import torch
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
import torchvision
import torch.nn.functional as F
import numpy as np

class Rotate(object):
    
    def __init__(self, theta):
        assert isinstance(theta, float)
        rot_mat = Rotate.get_rot_mat(theta)[None, ...].repeat(1,1,1)
        self.grid = F.affine_grid(rot_mat, torch.Size([1,1,28,28]))

    def __call__(self, sample):
        return F.grid_sample(sample.unsqueeze(dim=0), self.grid).squeeze(dim=0)
    
    @staticmethod
    def get_rot_mat(theta):
        theta = torch.tensor(theta)
        return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                             [torch.sin(theta), torch.cos(theta), 0]])

class AddNoise(object):
    
    def __init__(self, sigma = 0.2):
        assert isinstance(sigma, float)
        self.sigma = sigma

    def __call__(self, sample):
        return sample + self.sigma*torch.randn(sample.shape)
    

def get_mnist_dataset(train=True):
    batch_size = 60000 if train else 10000
    
    transform_rotate = torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   Rotate(np.pi/4),
                                   torchvision.transforms.Normalize(
                                     (0.1307,), (0.3081,))
                                 ])
    
    transform_original = torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                     (0.1307,), (0.3081,))
                                 ])
    
    transform_noise = torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                     (0.1307,), (0.3081,)),
                                   AddNoise(0.4)
                                 ])

    dataset_rotated = datasets.MNIST(root='./data', train=train, download=True, transform=transform_rotate)
    dataset_original = datasets.MNIST(root='./data', train=train, download=True, transform=transform_original)
    dataset_noisy = datasets.MNIST(root='./data', train=train, download=True, transform=transform_noise)

    dataloader_rotated = torch.utils.data.DataLoader(dataset_rotated, batch_size=batch_size,
                                          shuffle=False, num_workers=5)
    dataloader_noisy = torch.utils.data.DataLoader(dataset_noisy, batch_size=batch_size,
                                          shuffle=False, num_workers=5)
    dataloader_original = torch.utils.data.DataLoader(dataset_original, batch_size=batch_size,
                                          shuffle=False, num_workers=5)
    
    
    return torch.stack((next(iter(dataloader_original))[0],
                     next(iter(dataloader_rotated))[0],
                     next(iter(dataloader_noisy))[0]
                    )), next(iter(dataloader_original))[1]
