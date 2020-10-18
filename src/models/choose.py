import os
import torch
from models import pretrained, resnet


def get_model(name):
    """ Get model """

    print('[CHOOSE] Getting model: {}'.format(name))

    if name == 'pretrained-alexnet':

        return pretrained.AlexNet()

    elif name == 'pretrained-resnet18':

        return pretrained.ResNet18()

    elif name == 'resnet18':

        return resnet.resnet18()

    elif name == 'resnet34':

        return resnet.resnet34()

    elif name == 'resnet50':

        return resnet.resnet50()

    elif name == 'resnet101':

        return resnet.resnet101()

    elif name == 'resnet152':

        return resnet.resnet152()

    elif os.path.exists(name): # load model

        return load_model(name)

    raise ValueError('Model "{}" is not defined'.format(name))


def load_model(name):

    print('[CHOOSE] Loading model: {}'.format(name))

    # Get architecture
    model_architecture = name.split('arch_')[1].split('_')[0]

    # Call this function to get model
    model = get_model(model_architecture)

    # Load it
    model.load_state_dict(torch.load(name))

    return model