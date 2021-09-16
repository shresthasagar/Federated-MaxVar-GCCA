from dataset_mnist import get_mnist_dataset
train_set, train_classes = get_mnist_dataset(train=True, shuffle=True, normalize=False)
val_set, val_classes = get_mnist_dataset(train=False, total_data=1000, shuffle=True, normalize=False)

train_set = train_set.view(train_set.shape[0], train_set.shape[1], -1)
val_set = val_set.view(val_set.shape[0], val_set.shape[1], -1)

from collections import OrderedDict, namedtuple
from itertools import product
import os
from tqdm import tqdm, trange
from IPython.display import clear_output
import time
import torch.nn as nn
import pandas as pd
import torch
import matplotlib.pyplot as plt
from dgcca.models import g_step, MnistAEDGCCA, MnistAELinear, MnistAELinearBN
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
import torchvision
import numpy as np

from torch.optim.lr_scheduler import MultiStepLR

parser = argparse.ArgumentParser(description="MNIST DGCCA")

# Data
parser.add_argument('--model_dest', default='../trained_models/dgcca_mnist_cutemaxvar.model', help="Destination model path")
parser.add_argument('--random_seed', default=5555, help='')
parser.add_argument('--batch_size', default=1000, help='')
parser.add_argument('--lr', default=0.001, help='')
parser.add_argument('--device', default='cpu', help='')
parser.add_argument('--shuffle', default=True, help='')
parser.add_argument('--num_epochs', default=50, help='')
parser.add_argument('--latent_dim', default=10, help='')
parser.add_argument('--reg_weight', default=0.1, help='')

parser.add_argument('--compress', default=False, help='')
parser.add_argument('--compression_scheme', default='qsgd', help='')
parser.add_argument('--compress_downlink', default=False, help='')
parser.add_argument('--inner_epochs', default=10, help='')
parser.add_argument('--nbits', default=3, help='')

params = OrderedDict(
    lr = [0.001],
    batch_size = [1000],
    device = devices,
    shuffle = [True],
    num_workers = [5],
    manual_seed = [1265],
    loss_func = [nn.MSELoss],
    quant = [True],
    latent_dim = [10], 
    num_inner_epochs = [1]
)

args = vars(parser.parse_args([]))
pp.pprint(args)

# Set the random seed
torch.manual_seed(int(args['random_seed']))

# some special parameters
num_workers = 5
loss_func = nn.MSELoss

run_data = []
device = torch.device(args['device'])

# eval and train models and copy the train params to eval
dgcca = MnistAEDGCCA(output_size=args['latent_dim'], network=MnistAELinearBN)
dgcca = dgcca.to('cpu')

# Get train and validation dataset
train_views = list(train_set.to('cpu'))
val_views = list(val_set.to('cpu'))

optimizer = torch.optim.Adam(dgcca.parameters(), lr=args['lr'])
scheduler = MultiStepLR(optimizer, [30,70], gamma=0.8)
num_batches = len(train_views[0])//args['batch_size']

criterion = loss_func()

out = torch.stack(dgcca(train_views))
G_serv = g_step(out.clone().detach())  

M_serv = out.detach().clone()
G_client = G_serv.clone()

I = len(train_views)

for epoch in range(200):
    total_recons_loss = 0
    total_val_loss = 0
    batch_count = 0
    total_ae_loss = 0
    total_dgcca_loss = 0
    
    dgcca.train()
    dgcca.to(device)
    
    for _ in trange(args['inner_epochs']):
        for i in trange(num_batches):

            optimizer.zero_grad()
            batch = []

            # mini batch gradient
            batch = [view[(i*args['batch_size']):((i+1)*args['batch_size']), :].to(device) for view in train_views]            
            target = G_client[(i*args['batch_size']):((i+1)*args['batch_size']), :].to(device)

            latent = dgcca(batch)

            ae_loss = (args['latent_dim']/(2*28*28*target.shape[0]))*torch.norm(torch.stack(dgcca.decode(latent)) - torch.stack(batch))
            
            dgcca_loss = 1/2*torch.norm(torch.stack(latent)-target)/target.shape[0] 

            loss = dgcca_loss + reg_weight*ae_loss

            loss.backward()

            optimizer.step()

            total_recons_loss += loss.item()
            total_ae_loss += ae_loss.item()
            total_dgcca_loss += dgcca_loss.item()
            del batch, target, latent
    
    scheduler.step()
    
    dgcca.eval()
    dgcca.to('cpu')
    out = torch.stack(dgcca(train_views)).detach().clone()        
    if args['compress']:
        for i in range(I):
            diff = out[i] - M_serv[i]
            if args['compression_scheme'] == 'qsgd':
                quant = qsgd(diff, n_bits=args['nbits'])
            elif args['compression_scheme'] == 'deterministic':
                quant = deterministic_quantize(diff, n_bits=args['nbits'])
            M_serv[i] += quant
            M_serv[i] -= M_serv[i].mean(dim=0)
            del diff, quant

        G_serv = g_step(M_serv.clone().detach())
        if args['compress_downlink']:
            if args['compression_scheme'] == 'qsgd':
                G_client = G_client + qsgd(G_serv-G_client, n_bits=args['nbits'])
            elif args['compression_scheme'] == 'deterministic':
                G_client = G_client + deterministic_quantize(G_serv-G_client, n_bits=args['nbits'])
        else:
            G_client = G_serv.clone()
    else:
        out = out - out.mean(dim=1).unsqueeze(dim=1)
        M_serv = out.clone()
        G_serv = g_step(M_serv.clone().detach())  
        G_client = G_serv.clone() 
    del out
        
    # validation loss
    out_val = dgcca(val_views)
    out_val = torch.stack(out_val)    
    G_val = g_step(out_val.clone().detach())
    
    loss_val = 1/2*torch.norm(out_val-G_val)/G_val.shape[0]
    total_val_loss += loss_val.item()
    del out, G_val, out_val

    
    results = OrderedDict()
    results['epoch'] = epoch
    results['total_loss'] = total_recons_loss/(num_batches*args['inner_epochs'])
    results['ae_loss'] = total_ae_loss/(num_batches*args['inner_epochs'])
    results['dgcca_loss'] = total_dgcca_loss/(num_batches*args['inner_epochs'])
    results['val_fidelity'] = total_val_loss
    results['diff_norm'] = diff_norm
    results['lr'] = args['lr']
    
    run_data.append(results)
    df3 = pd.DataFrame.from_dict(run_data, orient='columns')
    clear_output(wait=True)
    display(df3)
    df3.to_pickle('plt/train_curve_mnist_dist.pkl')
    torch.save(dgcca, 'trained_models/dgcca_mnist_dist.model')
    