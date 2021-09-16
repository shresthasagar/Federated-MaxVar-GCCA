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
from dgcca.models import g_step, DeepGCCA
import pprint as pp
from dgcca.utils.compressor import qsgd
from dgcca.ehr.dataset_ehr import EhrDataset

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import random
import numpy as np
import math

if torch.cuda.is_available():
    devices = ['cuda']
else:
    devices = ['cpu']

parser = argparse.ArgumentParser(description="Toy 2d DGCCA")

# Data
parser.add_argument('--model_dest', default='trained_models/dgcca_ehr_altmaxvar.model', help="Destination model path")
parser.add_argument('--random_seed', default=5555, help='')
parser.add_argument('--compress', default=False, help='')
parser.add_argument('--compression_scheme', default='qsgd', help='')
parser.add_argument('--compress_downlink', default=True, help='')
parser.add_argument('--batch_size', default=200, help='')
parser.add_argument('--lr', default=0.0005, help='')
parser.add_argument('--device', default='cpu', help='')
parser.add_argument('--inner_epochs', default=10, help='')
parser.add_argument('--shuffle', default=True, help='')
parser.add_argument('--nbits', default=4, help='')
parser.add_argument('--num_epochs', default=30, help='')
parser.add_argument('--n_trials', default=10, help='')
parser.add_argument('--test_size', default=200, help='')


args = vars(parser.parse_args(args = []))
pp.pprint(args)

# Set the random seed
torch.manual_seed(int(args['random_seed']))
layer_sizes_list = 3*[[512, 256, 20]]
input_size_list = 3*[mat[0].shape[1]]

device = torch.device(args['device'])
num_workers = 5
loss_func = nn.MSELoss

num_batches = math.ceil(527/args['batch_size'])
acc_cute = np.zeros((args['n_trials'], args['num_epochs']))

ehr_dataset = EhrDataset(data_file='/scratch/sagar/Projects/federated_max_var_gcca/ehr/data/diag_med_frequent',
                            shuffle=True)
mat = ehr_dataset.x[:, :-args['test_size'], :]
test_data = ehr_dataset.x[:, -args['test_size']:, :]
test_views = [item.double() for item in test_data]
y_test = ehr_dataset.y[-args['test_size']:]
y_train = ehr_dataset.y[:-args['test_size']]

for trial_id in range(args['n_trials']):

    run_count = 0
    models = []
    run_data = []

    data_load_time = 0
    forward_time = 0

    indices = list(np.arange(mat[0].shape[0]))

    dgcca = DeepGCCA(layer_sizes_list, input_size_list)
    dgcca = dgcca.to(device)

    train_views = [item.double().to(device) for item in mat]
#     train_views = list(ehr_dataset.x.to(device))
    optimizer = torch.optim.Adam(dgcca.parameters(), lr=args['lr'])

    J = train_views[0].shape[0]

    criterion = loss_func()

    # init G
    dgcca.eval()
    out = dgcca(train_views)
    out = torch.stack(out)  
    out -= out.mean(dim=1).unsqueeze(dim=1)
    G = g_step(out.clone().detach())  
    M_serv = out.clone()
    M_diff = out.clone()
    dgcca.train()
    G_serv = G.clone().to(device)
    
    for epoch in trange(args['num_epochs']):
        total_recons_loss = 0
        total_val_loss = 0
        batch_count = 0

        for j in range(args['inner_epochs']):
            for i in range(num_batches):
                optimizer.zero_grad()
                batch = []
                
                batch_id = i%(num_batches)
                print(batch_id)
                # SGD
                try:
                    batch = [view[(batch_id*args['batch_size']):((batch_id+1)*args['batch_size']), :] for view in train_views]            
                    target = G_serv[(batch_id*args['batch_size']):((batch_id+1)*args['batch_size']), :]
                except:
                    batch = [view[(batch_id*args['batch_size']):, :] for view in train_views]            
                    target = G_serv[(batch_id*args['batch_size']):, :]
                
                # # SGD
                # rand_samp = random.sample(indices, args['batch_size'])
                # batch = [view[rand_samp,:] for view in train_views]
                # target = G[rand_samp, :]

                out = dgcca(batch)
                out = torch.stack(out)  

                loss = 1/2*torch.norm(out-target)/target.shape[0]

                loss.backward()
                optimizer.step()

                total_recons_loss += loss.item()
                del batch, target
        ## Update G
        dgcca.eval()
        out = dgcca(train_views)
        out = torch.stack(out)
        if args['compress']:
            for i in range(len(train_views)):
                M_diff[i] = out[i] - M_serv[i]
                max_val = M_diff[i].abs().max()

                if args['compression_scheme'] == 'qsgd':
                    M_quant = qsgd(M_diff[i], n_bits=args['nbits']).to(device)
                else:
                    M_quant = ((1/max_val)*M_diff[i]).round()*(max_val/1)

                M_serv[i] += M_quant
                M_serv[i] -= M_serv[i].mean(dim=0)
                del M_quant, max_val
            G = g_step(M_serv.clone().detach()) 
            G_serv = G_serv + qsgd(G-G_serv, n_bits=args['nbits']).to(device)
        else:
            out = out - out.mean(dim=1).unsqueeze(dim=1)
            G = g_step(out.clone().detach())  
        del out
        dgcca.train()

        # classification for distributed method
        clf = svm.SVC(kernel='rbf')
        clf.fit(G.to('cpu').numpy(), y_train)

        results = OrderedDict()
        results['epoch'] = epoch
        results['data_fidelity'] = total_recons_loss
        results['lr'] = args['lr']
        results['device'] = device

        dgcca = dgcca.to('cpu')
        out = torch.stack(dgcca(test_views))  
        out -= out.mean(dim=1).unsqueeze(dim=1)
        G_test = g_step(out.to('cpu').clone().detach())  

        results['class_acc'] = accuracy_score(y_test.numpy(), clf.predict(G_test.numpy()))
        dgcca = dgcca.to(device)
        acc_cute[trial_id, epoch] = results['class_acc']

        run_data.append(results)
        df_alt = pd.DataFrame.from_dict(run_data, orient='columns')
        clear_output(wait=True)
        display(df_alt)

        torch.save(dgcca, args['model_dest'])
        
