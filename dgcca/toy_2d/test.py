from collections import OrderedDict, namedtuple
from itertools import product
import argparse
from tqdm import tqdm, trange
from IPython.display import clear_output
import torch.nn as nn
import pandas as pd
import torch
import matplotlib.pyplot as plt
from dgcca.utils.run_manager import RunBuilder
from dgcca.utils.compressor import qsgd, deterministic_quantize
from dgcca.models import g_step, DeepGCCA
from dgcca.toy_2d.synth_data_toy_2d import create_synthData
import pprint as pp


parser = argparse.ArgumentParser(description="Toy 2d DGCCA")

# Data
parser.add_argument('--model_dest', default='../trained_models/dgcca_toy_cutemaxvar.model', help="Destination model path")
parser.add_argument('--random_seed', default=5555, help='')
parser.add_argument('--compress', default=False, help='')
parser.add_argument('--compression_scheme', default='qsgd', help='')
parser.add_argument('--compress_downlink', default=False, help='')
parser.add_argument('--batch_size', default=1000, help='')
parser.add_argument('--lr', default=0.001, help='')
parser.add_argument('--device', default='cpu', help='')
parser.add_argument('--inner_epochs', default=10, help='')
parser.add_argument('--shuffle', default=True, help='')
parser.add_argument('--nbits', default=3, help='')
parser.add_argument('--num_epochs', default=50, help='')

args = vars(parser.parse_args([]))
pp.pprint(args)

# Set the random seed
torch.manual_seed(int(args['random_seed']))

# some special parameters
layer_sizes_list = 3*[[128, 64, 2]]
input_size_list = 3*[2]
num_workers = 5
loss_func = nn.MSELoss

data_load_time = 0
forward_time = 0
run_data = []

device = torch.device(args['device'])

# eval and train models and copy the train params to eval
dgcca_train = DeepGCCA(layer_sizes_list, input_size_list)
dgcca_train = dgcca_train.to(device)
dgcca_eval = DeepGCCA(layer_sizes_list, input_size_list)
dgcca_eval.load_state_dict(dgcca_train.state_dict())

# Get train and validation dataset
train_views, classes = create_synthData(N=10000)
val_views, classes = create_synthData(N=200)
# shuffle the dataset
suffler = torch.randperm(10000)
train_views = [view[suffler].to(device) for view in train_views]
val_views = [view.to(device) for view in val_views]

optimizer = torch.optim.Adam(dgcca_train.parameters(), lr=args['lr'])
num_batches = len(train_views[0])//args['batch_size']

criterion = loss_func()
num_val_batches = len(val_views[0])//args['batch_size']

# init G
dgcca_eval.eval()
M_client = torch.stack(dgcca_eval(train_views))
G_server = g_step(M_client.clone().detach())  

M_serv = M_client.clone()
G_client = G_server.clone()

dgcca_train.train()

for epoch in range(args['num_epochs']):
    total_dgcca_loss = 0
    total_val_loss = 0
    batch_count = 0

    for j in range(args['inner_epochs']):
        for i in range(num_batches):
            optimizer.zero_grad()
            batch = []

            # SGD
            batch = [view[(i*args['batch_size']):((i+1)*args['batch_size']), :] for view in train_views]            
            target = G_client[(i*args['batch_size']):((i+1)*args['batch_size']), :]

            # full gradient
            # batch = train_views
            # target = G

            out = torch.stack(dgcca_train(batch))  

            loss = 1/2*torch.norm(out-target)/target.shape[0]

            loss.backward()
            optimizer.step()

            total_dgcca_loss += loss.item()

    ## Update G
    dgcca_eval.load_state_dict(dgcca_train.state_dict())
    M_client = dgcca_eval(train_views)
    M_client = torch.stack(M_client)

    if args['compress']:
        for i in range(len(train_views)):
            diff = M_client[i] - M_serv[i]
            max_val = diff.abs().max()
            if args['compression_scheme'] == 'qsgd':
                quant = qsgd(diff, n_bits=args['nbits'])
            else:
                quant = ((1/max_val)*diff).round()*(max_val/1)    
            M_serv[i] = M_serv[i] + quant
            M_serv[i] -= M_serv[i].mean(dim=0)
            del max_val, diff, quant
            
        G_serv = g_step(M_serv.clone().detach())
        if args['compress_downlink']:
            if args['compression_scheme'] == 'qsgd':
                G_client = G_client + qsgd(G_serv-G_client, n_bits=args['nbits'])
            else:
                # TODO: implement compression inside functions
                G_client = G_serv.clone()
        else:
            G_client = G_serv.clone()
    else:
        M_client = M_client - M_client.mean(dim=1).unsqueeze(dim=1)
        M_serv = M_client.clone()
        G_serv = g_step(M_serv.clone().detach())  
        G_client = G_serv.clone() 
    del M_client

    # validation loss
    out_val = torch.stack(dgcca_eval(val_views))
    G_val = g_step(out_val.clone().detach())
    loss_val = 1/2*torch.norm(out_val-G_val)/G_val.shape[0]
    total_val_loss = loss_val.item()

    results = OrderedDict()
    results['epoch'] = epoch
    results['data_fidelity'] = total_dgcca_loss/(num_batches*args['inner_epochs'])
    results['val_fidelity'] = total_val_loss
    results['batch_size'] = args['batch_size']
    results['lr'] = args['lr']
    results['device'] = args['device']
    
    run_data.append(results)
    df2 = pd.DataFrame.from_dict(run_data, orient='columns')
    clear_output(wait=True)
    display(df2)

    torch.save(dgcca_train, args['model_dest'])