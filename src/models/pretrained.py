import torch
import torch.nn as nn
from torchvision import models

class AlexNet(nn.Module): # inherits from the torch.nn.Module class
    
    def __init__(self):
        super().__init__()
        # Pre-trained AlexNet model
        self.pretrained_model = models.alexnet(pretrained=True)
        # Pooling layer
        self.pooling_layer = nn.AdaptiveAvgPool2d(1)
        # Dense layer that acts as a classification layer
        self.classifer = nn.Linear(256, 1)

    # In the forward method, we actually write the forward pass
    # i.e. the operations the network performs on the input until it computes the predictions.
    # this method recieves an input x of shape (1, s, 256, 256, 3) since we are dealing with batches of size 1
    def forward(self, x):
        # it removes the first dimension by "squeezing" the input and turning its shape to (s, 256, 256, 3)
        x = torch.squeeze(x, dim=0) 
        # now (s, 256, 256, 3) is a regular tensor shape that can be fed to an AlexNet 
        # which produces the features of shape (s, 256, 7, 7) afterwards
        features = self.pretrained_model.features(x)
        # the features are pooled which produces an output of shape (s, 256)
        pooled_features = self.pooling_layer(features)
        # the pooled features are flattened in a 256 dimension vector
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        flattened_features = torch.max(pooled_features, 0, keepdim=True)[0]
        # these features are finally fed to the classifier that outputs a scalar value. 
        # Note that we don't use a sigmoid activation here. Sigmoid is applied in the loss direction.
        output = self.classifer(flattened_features)
        return output


class ResNet18(nn.Module):
    
    def __init__(self, pretrained=True):
        super().__init__()
        # Pre-trained ResNet model
        self.pretrained_model = models.resnet18(pretrained=pretrained)
        num_features = self.pretrained_model.fc.in_features
        print('[PRE-TRAINED RESNET] Number of features: {}'.format(num_features))
        self.pretrained_model.fc = nn.Linear(num_features, 1)

    def forward(self, x):
        x = torch.squeeze(x, dim=0) 
        features = self.pretrained_model(x)
        output = torch.max(features, 0, keepdim=True)[0]
        return output