import torch
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader, Dataset
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
    

# def get_mnist_dataset(train=True, normalize=True, total_data=None):
#     if total_data is None:
#         batch_size = 60000 if train else 10000
#     else:
#         batch_size = total_data
        
#     if normalize:
#         mean = (0.1307, )
#         std = (0.3081, )
#     else:
#         mean = (0, )
#         std = (1, )

#     transform_rotate = torchvision.transforms.Compose([
#                                    torchvision.transforms.ToTensor(),
#                                    Rotate(np.pi/4),
#                                    torchvision.transforms.Normalize(
#                                      mean, std)
#                                  ])
    
#     transform_original = torchvision.transforms.Compose([
#                                    torchvision.transforms.ToTensor(),
#                                    torchvision.transforms.Normalize(
#                                      mean, std)
#                                  ])
    
#     transform_noise = torchvision.transforms.Compose([
#                                    torchvision.transforms.ToTensor(),
#                                    torchvision.transforms.Normalize(
#                                        mean, std),
#                                    AddNoise(0.4)
#                                  ])

#     dataset_rotated = datasets.MNIST(root='./data', train=train, download=True, transform=transform_rotate)
#     dataset_original = datasets.MNIST(root='./data', train=train, download=True, transform=transform_original)
#     dataset_noisy = datasets.MNIST(root='./data', train=train, download=True, transform=transform_noise)

#     dataloader_rotated = torch.utils.data.DataLoader(dataset_rotated, batch_size=batch_size,
#                                           shuffle=False, num_workers=5)
#     dataloader_noisy = torch.utils.data.DataLoader(dataset_noisy, batch_size=batch_size,
#                                           shuffle=False, num_workers=5)
#     dataloader_original = torch.utils.data.DataLoader(dataset_original, batch_size=batch_size,
#                                           shuffle=False, num_workers=5)

        
    
#     return torch.stack((next(iter(dataloader_original))[0],
#                      next(iter(dataloader_rotated))[0],
#                      next(iter(dataloader_noisy))[0]
#                     )), next(iter(dataloader_original))[1]

def get_flip_indices(y_sorted):
    flip_indices = [0]
    current_int = y_sorted[0]
    for i in range(len(y_sorted)):
        if current_int !=y_sorted[i]:
            current_int = y_sorted[i]
            flip_indices.append(i)         
    return flip_indices

def get_mnist_dataset(train=True, normalize=True, total_data=None, shuffle=False):
    if total_data is None:
        batch_size = 60000 if train else 10000
    else:
        batch_size = total_data
        
    if normalize:
        mean = (0.1307, )
        std = (0.3081, )
    else:
        mean = (0, )
        std = (1, )

    transform_rotate = torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   Rotate(np.pi/4),
                                   torchvision.transforms.Normalize(
                                     mean, std)
                                 ])
    
    transform_original = torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                     mean, std)
                                 ])
    
    transform_noise = torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       mean, std),
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

    x_test, y_test = torch.stack((next(iter(dataloader_original))[0],
                     next(iter(dataloader_rotated))[0],
                     next(iter(dataloader_noisy))[0]
                    )), next(iter(dataloader_original))[1]
    
    if shuffle:
        ## first shuffle only the samples corresponding to the same target
        ## The resulting dataset will be sorted by target
        a = np.argsort(y_test)
        x_test = x_test[:, a, :, :, :]
        y_test = y_test[a]

        flip_idx = get_flip_indices(y_test)
        flip_idx = flip_idx + [len(y_test)]

        for view in range(len(x_test)):
            randperms = torch.cat([flip_idx[i]+torch.randperm(flip_idx[i+1]-flip_idx[i]) for i in range(len(flip_idx)-1)])
            x_test[view] = x_test[view,randperms,:,:,:]
        
        ## Now shuffle the whole dataset 
        perm = torch.randperm(len(y_test))
        x_test = x_test[:,perm,:,:,:]
        y_test = y_test[perm]

    return x_test, y_test