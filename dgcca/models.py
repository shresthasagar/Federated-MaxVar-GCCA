"""
Author: Sagar Shrestha
Email: shressag@oregonstate.edu
"""
import torch
import torch.nn as nn

def g_step(M_list):
    """
    Solves the G optimization subproblem. 
    The solution is the SVD of the aggregated representations.
    """
    M = 0
    for i in range(len(M_list)):
        M += M_list[i]
    U, _, V = torch.svd(M, some=True)
    return torch.matmul(U, V.transpose(1,0))

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

class DummyModule(nn.Module):
    """
    Dummy module used for conditional module use
    """
    def __init__(self, num_params=None):
        super(DummyModule, self).__init__()

    def forward(self, x):
        return x

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

class AELinear(nn.Module):
    """
    Linear autoencoder Model 
    Use the arguments input_size, output_size, use_relu and use_batch_norm to control the design
    """
    def __init__(self, input_size=784, output_size=10, use_relu=True, use_batch_norm=False):
        super(AELinear, self).__init__()
        if use_relu:
            self.act = nn.ReLU
        else:
            self.act = nn.Sigmoid

        if use_batch_norm:
            self.norm = nn.BatchNorm1d
        else:
            self.norm = DummyModule

        self.encoder = nn.Sequential(
            Flatten(),
            nn.Linear(input_size, 1024),
            self.norm(1024),
            self.act(),
            
            nn.Linear(1024, 1024),
            self.norm(1024),
            self.act(),

            nn.Linear(1024, 1024),
            self.norm(1024),
            self.act(),

            nn.Linear(1024, output_size)
        )
        self.decoder = nn.Sequential(
            nn.Linear(output_size, 1024),
            self.norm(1024),
            self.act(),
            
            nn.Linear(1024, 1024),
            self.norm(1024),
            self.act(),

            nn.Linear(1024, 1024),
            self.norm(1024),
            self.act(),

            nn.Linear(1024, input_size),
            self.act()
        )

        self.encoder = nn.Sequential(
            Flatten(),
            nn.Linear(input_size, 512),
            self.norm(512),
            self.act(),
            
            nn.Linear(512, 256),
            self.norm(256),
            self.act(),

            nn.Linear(256 , output_size)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(output_size, 256),
            self.norm(256),
            self.act(),
            
            nn.Linear(256, 512),
            self.norm(512),
            self.act(),

            nn.Linear(512, input_size),
        )

    def forward(self, input):
        return self.encoder(input)

    def ae(self, input):
        return self.decoder(self.encoder(input))

class AE_DGCCA(nn.Module):
    """
    DGCCA based on autoencoder architecture
    
    For MNIST use the following configurations
        1. use_relu = False
        2. input_size = 784
        3. output_size = 10
    
    For EHR dataset use the following configurations
        1. use_relu = True
        2. input_size = 520 (depends upon the input, please check before using this number)
        3. output_size = 10
    """
    def __init__(self,
                    num_views=3,
                    input_size=784,
                    output_size=10, 
                    device=torch.device('cpu'), 
                    network = AELinear, 
                    use_relu=True,
                    use_batch_norm=True):
        super(AE_DGCCA, self).__init__()
        self.model_list = []
        for i in range(num_views):
            self.model_list.append(network(input_size=input_size, 
                                            output_size=output_size, 
                                            use_relu=use_relu,
                                            use_batch_norm=use_batch_norm))
        self.model_list = nn.ModuleList(self.model_list)

    def forward(self, x_list):
        """
        x_%  are the vectors which need to be made correlated
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



class DGCCAClassifierToy(nn.Module):
    """
    Train a NN classifier for each view and the common representation in order to evaluate the trained DGCCA model
    """
    def __init__(self, num_views):
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


class MnistDGCCA(nn.Module):
    """
    DGCA mdel with Mnist dataset
    """
    def __init__(self, num_views=3, output_size=10, device=torch.device('cpu')):
        class MnistNet(nn.Module):
            """
            MNIST DGCCA ConvNet architecture and BN
            """
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

        super(MnistDGCCA, self).__init__()
        self.model_list = []
        for i in range(num_views):
            self.model_list.append(MnistNet(output_size=output_size))
        self.model_list = nn.ModuleList(self.model_list)
        

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


class MnistAELinear(nn.Module):
    """
    Linear Mnist Model without batch Normalization
    Use it only with [0,1] normalized data because it contains sigmoid activation in the output
    This is done in order to reproduce DCCAE paper, "On Deep Multi-View Representation Learning", Weiran et al, 2015 
    """
    def __init__(self, output_size=10):
        super(MnistAELinear, self).__init__()
        ndf = 16
        nc = 1
        in_size = 784
        self.encoder = nn.Sequential(
            Flatten(),
            nn.Linear(in_size, 1024),
            nn.Sigmoid(),
            
            nn.Linear(1024, 1024),
            nn.Sigmoid(),

            nn.Linear(1024, 1024),
            nn.Sigmoid(),

            nn.Linear(1024, output_size)
        )

        self.decoder = nn.Sequential(
            nn.Linear(output_size, 1024),
            nn.Sigmoid(),
            
            nn.Linear(1024, 1024),
            nn.Sigmoid(),

            nn.Linear(1024, 1024),
            nn.Sigmoid(),

            nn.Linear(1024, in_size),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.encoder(input)

    def ae(self, input):
        return self.decoder(self.encoder(input))


#TODO: Delete this and use the AELinearBN with input size argument of 784
class MnistAELinearBN(nn.Module):
    """
    Linear Mnist Model with batch Normalization
    Use it only with [0,1] normalized data because it contains sigmoid activation in the output
    This is done in order to reproduce DCCAE paper, "On Deep Multi-View Representation Learning", Weiran et al, 2015 
    """
    def __init__(self, output_size=10):
        super(MnistAELinearBN, self).__init__()
        ndf = 16
        nc = 1
        in_size = 784
        self.encoder = nn.Sequential(
            Flatten(),
            nn.Linear(in_size, 1024),
            nn.BatchNorm1d(1024),
            nn.Sigmoid(),
            
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.Sigmoid(),

            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.Sigmoid(),

            nn.Linear(1024, output_size)
        )

        self.decoder = nn.Sequential(
            nn.Linear(output_size, 1024),
            nn.BatchNorm1d(1024),
            nn.Sigmoid(),
            
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.Sigmoid(),

            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.Sigmoid(),

            nn.Linear(1024, in_size),
            nn.Sigmoid()
        )
    def forward(self, input):
        return self.encoder(input)

    def ae(self, input):
        return self.decoder(self.encoder(input))

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

#TODO: Rename it to MnistConvBN
class MnistAutoencoderNoBN(nn.Module):
    def __init__(self, output_size=10):
        super(MnistAutoencoderNoBN, self).__init__()
        ndf = 16
        nc = 1
        self.encoder = nn.Sequential(
            # input is (nc) x 51 x 51
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.SELU(),

            # state size. (ndf) x 25 x 25
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.SELU(),
            
            # state size. (ndf*2) x 12 x 12
            nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False),
            nn.SELU(),
            
            # state size. (ndf*4) x 6 x 6
            nn.Conv2d(ndf * 4, ndf * 4, 4, 2, 1, bias=False),
            nn.SELU(),

            Flatten(), 
            nn.Linear(256, output_size)
        )

        self.decoder = nn.Sequential(
            nn.Linear(output_size, 256),

            UnFlatten((64,2,2)),
            # input is Z, going into a convolution
            nn.ConvTranspose2d(ndf*4, ndf*4, 4, 2, 1, bias=False),
            nn.SELU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ndf*4, ndf * 2, 3, 2, 1, bias=False),
            nn.SELU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ndf * 2, ndf* 1, 4, 2, 1, bias=False),
            nn.SELU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ndf, nc, 4, 2, 1, bias=False),
            
        )

    def forward(self, input):
        return self.encoder(input)

    def ae(self, input):
        return self.decoder(self.encoder(input))


class MnistAEDGCCA(nn.Module):
    def __init__(self, num_views=3, output_size=10, device=torch.device('cpu'), network=MnistAutoencoder):
        super(MnistAEDGCCA, self).__init__()
        self.model_list = []
        for i in range(num_views):
            self.model_list.append(network(output_size=output_size))
        self.model_list = nn.ModuleList(self.model_list)
        

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


class DGCCAClassifierMnist(nn.Module):
    """
    Train a NN classifier for each view and the common representation in order to evaluate the trained DGCCA model
    """
    def __init__(self, num_views):
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
