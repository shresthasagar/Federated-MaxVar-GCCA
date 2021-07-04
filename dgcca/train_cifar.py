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
from utils.run_manager import RunBuilder
from models import g_step, CifarDGCCA
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
import torchvision

class RunBuilder():
    @staticmethod
    def get_runs(params):

        Run = namedtuple('Run', params.keys())

        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))

        return runs

if torch.cuda.is_available():
    devices = ['cpu']
else:
    devices = ['cpu']
print('starting')


params = OrderedDict(
    lr = [0.001],
    batch_size = [1000],
    device = devices,
    shuffle = [True],
    num_workers = [5],
    manual_seed = [1265],
    loss_func = [nn.MSELoss],
    quant = [True]
)

# layer_sizes_list = 3*[[128, 64, 2]]
# input_size_list = 3*[2]


run_count = 0
models = []


run_data = []

data_load_time = 0
forward_time = 0


for run in RunBuilder.get_runs(params):
#     torch.cuda.set_device(run.device)
    
    run_count += 1
    device = torch.device(run.device)
    
    dgcca = CifarDGCCA()
    dgcca = dgcca.to(device)

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) #torchvision.transforms.Normalize(mean, std)

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=50000,
                                              shuffle=False, num_workers=5)

    test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=5000,
                                             shuffle=False, num_workers=5)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    train_loader = iter(train_loader)
    test_loader = iter(test_loader)
    
    train_views, _ = next(train_loader)
    val_views, _ = next(test_loader)
    
    train_views = [train_views[:,i,:,:] for i in range(train_views.shape[1])]
    train_views = [view.to(device).unsqueeze(dim=1) for view in train_views]
    
    val_views = [val_views[:,i,:,:] for i in range(val_views.shape[1])]
    val_views = [view.to(device).unsqueeze(dim=1) for view in val_views]
    
    optimizer = torch.optim.Adam(dgcca.parameters(), lr=run.lr)
    num_batches = len(train_views[0])//run.batch_size
    
    criterion = run.loss_func()
    num_val_batches = len(val_views[0])//1000
    
    out = torch.stack(dgcca(train_views))
#     out = torch.stack(out)  
    G = g_step(out.clone().detach())  
    
    M_serv = out.clone()
    
    I = len(train_views)
    
    for epoch in range(100):
        total_recons_loss = 0
        total_val_loss = 0
        batch_count = 0
        
        results = OrderedDict()
        results['run_count'] = run_count
        results['epoch'] = epoch
        results['data_fidelity'] = total_recons_loss/num_batches
        results['val_fidelity'] = total_recons_loss/num_batches
        
        results['batch_size'] = run.batch_size
        results['lr'] = run.lr
        results['device'] = run.device

        df3 = pd.DataFrame.from_dict(run_data, orient='columns')
        
        for i in range(num_batches):
            optimizer.zero_grad()
            batch = []
            
            # mini batch gradient
            batch = [view[(i*run.batch_size):((i+1)*run.batch_size), :] for view in train_views]            
            target = G[(i*run.batch_size):((i+1)*run.batch_size), :]
            
            # full gradient
#             batch = train_views
#             target = G
#             print(batch[0].shape)
#             out = dgcca(batch)
#             out = torch.stack(out)  

            loss = 1/2*torch.norm(torch.stack(dgcca(batch))-target)/target.shape[0]
            
            loss.backward()
            optimizer.step()
            
            total_recons_loss += loss.item()
            
            del batch, target
            
        ## initialize G
#         out = dgcca(train_views)
        out = torch.stack(dgcca(train_views))
        
        if run.quant:
            for i in range(I):
                diff = out[i] - M_serv[i]
                max_val = diff.abs().max()
                quant = ((1/max_val)*diff[i]).round()*(max_val/1)
                M_serv[i] = M_serv[i] - quant
#                 del max_val, diff, quant
            G = g_step(M_serv.clone().detach())
        else:
#             pass
            G = g_step(out.clone().detach())
            
        # validation loss
        out_val = dgcca(val_views)
        out_val = torch.stack(out_val)
        
        G_val = g_step(out_val.clone().detach())
        
        loss_val = criterion(out_val, G_val)
        total_val_loss += loss_val.item()
        
        del out, G_val, out_val
        
        results = OrderedDict()
        results['run_count'] = run_count
        results['epoch'] = epoch
        results['data_fidelity'] = total_recons_loss/num_batches/10
        results['val_fidelity'] = total_val_loss
        results['batch_size'] = run.batch_size
        results['lr'] = run.lr
        results['device'] = run.device
        
        run_data.append(results)
        df3 = pd.DataFrame.from_dict(run_data, orient='columns')
#         clear_output(wait=True)
# #         show_latent()
#         display(df3)
        print(epoch, results['data_fidelity'])
#             m.track_loss(G_adv_loss=losses['beta_kl-divergence'], G_mse_loss=losses[''], D_real_loss=total_D_real, D_fake_loss=total_D_fake, D_real_count=real_count, D_fake_count=fake_count)
#         print(epoch, "total_Gloss:",total_Gloss, "total_Dloss:",total_Dloss, "mse:",total_mse_loss, "adv: ", total_adv_loss)           
#         m.end_epoch()
        torch.save(dgcca, 'trained_models/dgcca_cifar10_fed1.model')