import torchvision
from .resnet_spiking import *


def get_resnet(name, pretrained=False):
    resnets = {
        "resnet18": torchvision.models.resnet18(pretrained=pretrained),
        "resnet50": torchvision.models.resnet50(pretrained=pretrained),
    }
    if name not in resnets.keys():
        raise KeyError(f"{name} is not a valid ResNet version")
    return resnets[name]

def get_resnet_spiking(name, device, batch_size):
    return RESNET_SNN_STDB(resnet_name=name, activation='STDB',
                           labels=1000, timesteps=100, leak=1.0,
                           default_threshold=1.0, alpha=0.3,
                           beta=0.01, dropout=0.3, device=device, input_shape=[batch_size, 3, 224, 224])