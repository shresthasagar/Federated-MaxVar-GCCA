import torch
import matplotlib.pyplot as plt
from utils.synth_data_toy_2d import create_synthData
from models import g_step

def plot_latent(dgcca_vanilla_path = 'trained_models/dgcca1.model', dgcca_fed_path='trained_models/dgcca_federated1.model', device='cuda'):
    dgcca_vanilla = torch.load(dgcca_vanilla_path)
    dgcca_fed = torch.load(dgcca_fed_path)

    train_views, classes = create_synthData(N=1000)
    val_views, classes = create_synthData(N=200)

    train_views = [view.to(device) for view in train_views]
    val_views = [view.to(device) for view in val_views]

    out1 = dgcca_vanilla(train_views)
    out1 = [a.to('cpu').detach() for a in out1]


    out1 = dgcca_vanilla(train_views)
    G1 = g_step(torch.stack(out1).clone().detach())
    out1 = [a.to('cpu').detach() for a in out1]

    out2 = dgcca_fed(train_views)
    G2 = g_step(torch.stack(out2).clone().detach())
    out2 = [a.to('cpu').detach() for a in out2]

    G1 = G1.detach().to('cpu')
    G2 = G2.detach().to('cpu')

    target = [t.to('cpu').detach() for t in train_views]

    fig, axes = plt.subplots(4,3, figsize=(10,10))

    axes[0,0].set_title('Data')
    axes[0,0].scatter(target[0][:,0], target[0][:,1], c=500*[1]+500*[2])
    axes[0,0].set_ylabel('View1')
    axes[1,0].scatter(target[1][:,0], target[1][:,1], c=500*[1]+500*[2])
    axes[1,0].set_ylabel('View2')
    axes[2,0].scatter(target[2][:,0], target[2][:,1], c=500*[1]+500*[2])
    axes[2,0].set_ylabel('View3')
    axes[3,0].set_ylabel('Latent Representation (G)')

    axes[0,1].set_title("Vanilla DGCCA")
    axes[0,1].scatter(out1[0][:,0], out1[0][:,1], c=500*[1]+500*[2])
    axes[1,1].scatter(out1[1][:,0], out1[1][:,1], c=500*[1]+500*[2])
    axes[2,1].scatter(out1[2][:,0], out1[2][:,1], c=500*[1]+500*[2])
    axes[3,1].scatter(G1[:,0], G1[:,1], c=500*[1]+500*[2])

    axes[0,2].set_title("Federated DGCCA, 2 bits per scalar")
    axes[0,2].scatter(out2[0][:,0], out2[0][:,1], c=500*[1]+500*[2])
    axes[1,2].scatter(out2[1][:,0], out2[1][:,1], c=500*[1]+500*[2])
    axes[2,2].scatter(out2[2][:,0], out2[2][:,1], c=500*[1]+500*[2])
    axes[3,2].scatter(G2[:,0], G2[:,1], c=500*[1]+500*[2])
