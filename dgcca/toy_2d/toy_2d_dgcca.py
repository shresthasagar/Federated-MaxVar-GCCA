from collections import OrderedDict, namedtuple
from itertools import product
import argparse
from tqdm import tqdm, trange
from IPython.display import clear_output
import torch.nn as nn
import pandas as pd
import torch
import matplotlib.pyplot as plt
from utils.run_manager import RunBuilder
from models import g_step, DeepGCCA
from utils.synth_data_toy_2d import create_synthData
import pprint as pp

if torch.cuda.is_available():
    devices = ['cuda']
else:
    devices = ['cpu']

parser = argparse.ArgumentParser(description="Toy 2d DGCCA")

# Data
parser.add_argument('--model_dest', default='trained_models/dgcca_toy_2d1.model', help="Destination model path")
parser.add_argument('--random_seed', default=23423, help='')

args = vars(parser.parse_args())
pp.pprint(args)

# Set the random seed
torch.manual_seed(int(args['random_seed']))

params = OrderedDict(
    lr = [0.001],
    batch_size = [1000],
    device = devices,
    shuffle = [True],
    num_workers = [5],
    manual_seed = [1265],
    loss_func = [nn.MSELoss],
    inner_epochs = [50],
    quant = [True]
)

layer_sizes_list = 3*[[128, 64, 2]]
input_size_list = 3*[2]

run_count = 0
models = []

run_data = []

data_load_time = 0
forward_time = 0


for run in RunBuilder.get_runs(params):
  
    run_count += 1
    device = torch.device(run.device)
    
    dgcca = DeepGCCA(layer_sizes_list, input_size_list)
    dgcca = dgcca.to(device)
    
    train_views, classes = create_synthData(N=10000)
    val_views, classes = create_synthData(N=200)
    suffler = torch.randperm(10000)
    
    train_views = [view[suffler].to(device) for view in train_views]
    val_views = [view.to(device) for view in val_views]
    
    optimizer = torch.optim.Adam(dgcca.parameters(), lr=run.lr)
    num_batches = len(train_views[0])//run.batch_size
    
    criterion = run.loss_func()
    num_val_batches = len(val_views[0])//run.batch_size
    
    # init G
    dgcca.eval()
    out = dgcca(train_views)
    out = torch.stack(out)  
    G = g_step(out.clone().detach())  
    M_serv = out.clone()
    M_diff = out.clone()
    dgcca.train()
    
    for epoch in range(50):
        total_recons_loss = 0
        total_val_loss = 0
        batch_count = 0
        
        for j in range(run.inner_epochs):
            for i in range(num_batches):
                optimizer.zero_grad()
                batch = []
                
                # SGD
                batch = [view[(i*run.batch_size):((i+1)*run.batch_size), :] for view in train_views]            
                target = G[(i*run.batch_size):((i+1)*run.batch_size), :]

                # full gradient
                # batch = train_views
                # target = G

                out = dgcca(batch)
                out = torch.stack(out)  
                
                loss = 1/2*torch.norm(out-target)/target.shape[0]
                
                loss.backward()
                optimizer.step()
                
                total_recons_loss += loss.item()
                
        ## Update G
        dgcca.eval()
        out = dgcca(train_views)
        out = torch.stack(out)
        
        if run.quant:
            for i in range(len(train_views)):
                M_diff[i] = out[i] - M_serv[i]
                max_val = M_diff[i].abs().max()
                M_quant = ((1/max_val)*M_diff[i]).round()*(max_val/1)
                M_serv[i] += M_quant
            G = g_step(M_serv.clone().detach())          
        else:
            G = g_step(out.clone().detach())  
            
        
        # validation loss
        out_val = dgcca(val_views)
        out_val = torch.stack(out_val)
        G_val = g_step(out_val.clone().detach())
        loss_val = 1/2*torch.norm(out_val - G_val)/G_val.shape[0]
        total_val_loss = loss_val.item()

        dgcca.train()
        
        results = OrderedDict()
        results['run_count'] = run_count
        results['epoch'] = epoch
        results['data_fidelity'] = total_recons_loss/(num_batches*run.inner_epochs)
        results['val_fidelity'] = total_val_loss
        results['batch_size'] = run.batch_size
        results['lr'] = run.lr
        results['device'] = run.device
        
        run_data.append(results)
        # df2 = pd.DataFrame.from_dict(run_data, orient='columns')
        # clear_output(wait=True)
        # show_latent()
        # display(df2)

        torch.save(dgcca, args['model_dest'])