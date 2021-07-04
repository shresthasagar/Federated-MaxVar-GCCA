import torch
import torch.nn as nn

from loss_objectives import GCCA_loss

## Helpder Classes 
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def __init__(self, target_shape):
        super().__init__()
        self.target_shape = target_shape
        
    def forward(self, input):
        return torch.reshape(input, (input.size(0),*self.target_shape))


## Toy DGCCA for 2-D points
class MlpNet(nn.Module):
    def __init__(self, layer_sizes, input_size):
        super(MlpNet, self).__init__()
        layers = []
        layer_sizes = [input_size] + layer_sizes
        for l_id in range(len(layer_sizes) - 1):
            if l_id == len(layer_sizes) - 2:
                layers.append(nn.Sequential(
                    nn.Linear(layer_sizes[l_id], layer_sizes[l_id + 1])
                ))
            else:
                layers.append(nn.Sequential(
                    nn.Linear(layer_sizes[l_id], layer_sizes[l_id + 1]),
                    nn.BatchNorm1d(num_features=layer_sizes[l_id + 1]),
                    nn.ReLU(),
                ))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class DeepGCCA(nn.Module):
    def __init__(self, layer_sizes_list, input_size_list, device=torch.device('cpu')):
        super(DeepGCCA, self).__init__()
        self.model_list = []
        for i in range(len(layer_sizes_list)):
            self.model_list.append(MlpNet(layer_sizes_list[i], input_size_list[i]).double())
        self.model_list = nn.ModuleList(self.model_list)
        self.loss = GCCA_loss


    def forward(self, x_list):
        """

        x_%  are the vectors needs to be make correlated
        dim = [batch_size, features]

        """
        # feature * batch_size
        output_list = []
        for x, model in zip(x_list, self.model_list):
            output_list.append(model(x))

        return output_list

## Classifier to evaluate Toy DGCCA
class MLPClassifierToy(nn.Module):
    def __init__(self):
        super(MLPClassifierToy, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(2,64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64,64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.Linear(64,1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x)

class DGCCAClassifierToy(nn.Module):
    def __init__(self, num_views):
        super(DGCCAClassifierToy, self).__init__()
        
        self.model_list = []
        for i in range(num_views + 1):
            self.model_list.append(MLPClassifierToy())
        self.model_list = nn.ModuleList(self.model_list)
        self.criterion = nn.BCELoss()

    def forward(self, x_list):
        """

        x_%  are the vectors needs to be make correlated
        dim = [batch_size, features]

        """
        # feature * batch_size
        output_list = []
        for x, model in zip(x_list, self.model_list):
            output_list.append(model(x))

        return output_list

    def loss_backward(self, output_list, target_list):
        loss = []
        for i, (out, target) in enumerate(zip(output_list, target_list)):
            loss.append(self.criterion(out, target))
            loss[i].backward()
        loss = [i.item() for i in loss]
        return loss

    def loss(self, output_list, target_list):
        loss = []
        for i, (out, target) in enumerate(zip(output_list, target_list)):
            loss.append(self.criterion(out, target))
        loss = [i.item() for i in loss]
        return loss
    

## DGCCA on CelebA dataset
class CelebaNet(nn.Module):
    def __init__(self, output_size=10):
        super(CelebaNet, self).__init__()
        ndf = 16
        nc = 1
        self.main = nn.Sequential(
            # input is (nc) x 51 x 51
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.SELU(),

            # state size. (ndf) x 25 x 25
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.SELU(),
            
            # state size. (ndf*2) x 12 x 12
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.SELU(),
            
            # state size. (ndf*4) x 6 x 6
            nn.Conv2d(ndf * 4, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.SELU(),
            
            # state size. (ndf*8) x 3 x 3
            nn.Conv2d(ndf * 4, ndf * 4, 4, 2, 1, bias=False),
            nn.SELU(),

            Flatten(), 
            nn.Linear(256, output_size)
        )

    def forward(self, input):
        return self.main(input)

class CelebaDGCCA(nn.Module):
    def __init__(self, num_views=3, output_size=10, device=torch.device('cpu')):
        super(CelebaDGCCA, self).__init__()
        self.model_list = []
        for i in range(num_views):
            self.model_list.append(CelebaNet(output_size=output_size))
        self.model_list = nn.ModuleList(self.model_list)
        self.loss = GCCA_loss


    def forward(self, x_list):
        """

        x_%  are the vectors needs to be make correlated
        dim = [batch_size, features]

        """
        # feature * batch_size
        output_list = []
        for x, model in zip(x_list, self.model_list):
            output_list.append(model(x))

        return output_list


## DGCCA on CIFAR10 dataset
class CifarNet(nn.Module):
    def __init__(self, output_size=10):
        super(CifarNet, self).__init__()
        ndf = 16
        nc = 1
        self.main = nn.Sequential(
            # input is (nc) x 51 x 51
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.SELU(),

            # state size. (ndf) x 25 x 25
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.SELU(),
            
            # state size. (ndf*2) x 12 x 12
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.SELU(),
            
            # state size. (ndf*8) x 3 x 3
            nn.Conv2d(ndf * 4, ndf * 4, 4, 2, 1, bias=False),
            nn.SELU(),

            Flatten(), 
            nn.Linear(256, output_size)
        )

    def forward(self, input):
        return self.main(input)

class CifarDGCCA(nn.Module):
    def __init__(self, num_views=3, output_size=10, device=torch.device('cpu')):
        super(CifarDGCCA, self).__init__()
        self.model_list = []
        for i in range(num_views):
            self.model_list.append(CifarNet(output_size=output_size))
        self.model_list = nn.ModuleList(self.model_list)
        self.loss = GCCA_loss


    def forward(self, x_list):
        """

        x_%  are the vectors needs to be make correlated
        dim = [batch_size, features]

        """
        # feature * batch_size
        output_list = []
        for x, model in zip(x_list, self.model_list):
            output_list.append(model(x))

        return output_list


## CIFAR10 with labels as one of the views

class MLPLabeledView(nn.Module):
    """
    Using labels as one of the views for CIFAR 10 dataset
    """
    def __init__(self, output_size=10):
        super(MLPLabeledView, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(10,128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128,64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.Linear(64, output_size, bias=True),
        )

    def forward(self, x):
        return self.layers(x)

class CifarDGCCALabeled(nn.Module):
    def __init__(self, num_views=3, output_size=10, device=torch.device('cpu')):
        super(CifarDGCCALabeled, self).__init__()
        self.model_list = []
        for i in range(num_views):
            self.model_list.append(CifarNet(output_size=output_size))
        self.model_list.append(MLPLabeledView(output_size=output_size))
        self.model_list = nn.ModuleList(self.model_list)
        self.loss = GCCA_loss


    def forward(self, x_list):
        """

        x_%  are the vectors needs to be make correlated
        dim = [batch_size, features]

        """
        # feature * batch_size
        output_list = []
        for x, model in zip(x_list, self.model_list):
            output_list.append(model(x))

        return output_list

## MNIST DGCCA Regular architecture

class MnistNet(nn.Module):
    def __init__(self, output_size=10):
        super(MnistNet, self).__init__()
        ndf = 16
        nc = 1
        self.main = nn.Sequential(
            # input is (nc) x 51 x 51
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.SELU(),

            # state size. (ndf) x 25 x 25
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.SELU(),
            
            # state size. (ndf*2) x 12 x 12
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.SELU(),
            
            # state size. (ndf*4) x 6 x 6
            nn.Conv2d(ndf * 4, ndf * 4, 4, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.SELU(),

            Flatten(), 
            nn.Linear(256, output_size)
        )

    def forward(self, input):

        return self.main(input)


class MnistDGCCA(nn.Module):
    def __init__(self, num_views=3, output_size=10, device=torch.device('cpu')):
        super(MnistDGCCA, self).__init__()
        self.model_list = []
        for i in range(num_views):
            self.model_list.append(MnistNet(output_size=output_size))
        self.model_list = nn.ModuleList(self.model_list)
        self.loss = GCCA_loss

    def forward(self, x_list):
        """
        x_%  are the vectors needs to be make correlated
        dim = [batch_size, features]
        """
        # feature * batch_size
        output_list = []
        for x, model in zip(x_list, self.model_list):
            output_list.append(model(x))
        return output_list

## MNIST DGCCA with autoencoder architecture for promoting clustering
class MnistAutoencoder(nn.Module):
    def __init__(self, output_size=10):
        super(MnistAutoencoder, self).__init__()
        ndf = 16
        nc = 1
        self.encoder = nn.Sequential(
            # input is (nc) x 51 x 51
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.SELU(),

            # state size. (ndf) x 25 x 25
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.SELU(),
            
            # state size. (ndf*2) x 12 x 12
            nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.SELU(),
            
            # state size. (ndf*4) x 6 x 6
            nn.Conv2d(ndf * 4, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.SELU(),

            Flatten(), 
            nn.Linear(256, output_size)
        )

        self.decoder = nn.Sequential(
            nn.Linear(output_size, 256),

            UnFlatten((64,2,2)),
            # input is Z, going into a convolution
            nn.ConvTranspose2d(ndf*4, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ndf*4, ndf * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ndf * 2, ndf* 1, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ndf, nc, 4, 2, 1, bias=False),
            
        )

    def forward(self, input):
        return self.encoder(input)

    def ae(self, input):
        return self.decoder(self.encoder(input))

class MnistClassifier(nn.Module):
    def __init__(self):
        super(MnistClassifier, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(10,128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128,128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Linear(128,10),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x)

class DGCCAClassifierMnist(nn.Module):
    def __init__(self, num_views):
        super(DGCCAClassifierMnist, self).__init__()
        
        self.model_list = []
        for i in range(num_views + 1):
            self.model_list.append(MnistClassifier())
        self.model_list = nn.ModuleList(self.model_list)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x_list):
        """

        x_%  are the vectors needs to be make correlated
        dim = [batch_size, features]

        """
        # feature * batch_size
        output_list = []
        for x, model in zip(x_list, self.model_list):
            output_list.append(model(x))

        return output_list

    def loss_backward(self, output_list, target_list):
        loss = []
        for i, (out, target) in enumerate(zip(output_list, target_list)):
            loss.append(self.criterion(out, target))
            loss[i].backward()
        loss = [i.item() for i in loss]
        return loss

    def loss(self, output_list, target_list):
        loss = []
        for i, (out, target) in enumerate(zip(output_list, target_list)):
            loss.append(self.criterion(out, target))
        loss = [i.item() for i in loss]
        return loss

    def accuracy(self, output_list, target_list):
        correct = []
        total = []
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for i, (out, target) in enumerate(zip(output_list, target_list)):
                _, predicted = torch.max(out, 1)
                total.append(out.size(0))
                correct.append((predicted == target).sum().item())
        return [correct[i]/total[i] for i in range(len(total))]

class MnistAEDGCCA(nn.Module):
    def __init__(self, num_views=3, output_size=10, device=torch.device('cpu')):
        super(MnistAEDGCCA, self).__init__()
        self.model_list = []
        for i in range(num_views):
            self.model_list.append(MnistAutoencoder(output_size=output_size))
        self.model_list = nn.ModuleList(self.model_list)
        self.loss = GCCA_loss

    def forward(self, x_list):
        """
        x_%  are the vectors needs to be make correlated
        dim = [batch_size, features]
        Runs through the encoder of the autoencoder
        """
        # feature * batch_size
        output_list = []
        for x, model in zip(x_list, self.model_list):
            output_list.append(model(x))
        return output_list

    def ae(self, x_list):
        """
        Run through encoder and decoder
        """
        output_list = []
        for x, model in zip(x_list, self.model_list):
            output_list.append(model.ae(x))
        return output_list

    def decode(self, x_list):
        """
        Runs through the decoder of the autoencoder
        """
        output_list = []
        for x, model in zip(x_list, self.model_list):
            output_list.append(model.decoder(x))
        return output_list
        

def g_step(M_list):
    M = 0
    for i in range(len(M_list)):
        M += M_list[i]
    U, _, V = torch.svd(M, some=True)
    return torch.matmul(U, V.transpose(1,0))