# SPIKING VERSION
from .spike_layer import *
from math import sqrt
import torch.nn.functional as FS


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride, timestep, modified=True):
        super(BasicBlock, self).__init__()

        self.timestep = timestep
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = tdBatchNorm(nn.BatchNorm2d(planes))
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = tdBatchNorm(nn.BatchNorm2d(planes))
        self.spike_func = MLF_unit(self.timestep)
        self.shortcut = nn.Sequential()
        self.modified = modified

        if stride != 1 or in_planes != planes:
            if self.modified:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                    tdBatchNorm(nn.BatchNorm2d(planes)),
                    MLF_unit(self.timestep)
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                    tdBatchNorm(nn.BatchNorm2d(planes)),
                )

    def forward(self, x):
        out = self.spike_func(self.bn1(self.conv1(x)))

        out = self.bn2(self.conv2(out))

        if self.modified:
            out = self.spike_func(out)
            out += self.shortcut(x)         # Equivalent to union of all spikes
        else:
            out += self.shortcut(x)
            out = self.spike_func(out)
        return out


class BLock_Layer(nn.Module):
    def __init__(self, block, in_planes, planes, num_block, timestep, downsample, modified):
        super(BLock_Layer, self).__init__()

        layers = []
        if downsample:
            layers.append(block(in_planes, planes, 2, timestep, modified))
        else:
            layers.append(block(in_planes, planes, 1, timestep, modified))
        for _ in range(1, num_block):
            layers.append(block(planes, planes, 1, timestep, modified))
        self.execute = nn.Sequential(*layers)

    def forward(self, x):
        return self.execute(x)


class ResNet(nn.Module):
    """ Establish ResNet.
     Spiking DS-ResNet with “modified=True.”
     Spiking ResNet with “modified=False.”
     """
    def __init__(self, block, num_block_layers, timestep, num_classes=10):
        super(ResNet, self).__init__()

        self.timestep = timestep
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = tdBatchNorm(nn.BatchNorm2d(64))
        self.layer1 = BLock_Layer(block, 64, 64, num_block_layers[0], self.timestep, False, modified=True)
        self.layer2 = BLock_Layer(block, 64, 128, num_block_layers[1], self.timestep, True, modified=True)
        self.layer3 = BLock_Layer(block, 128, 256, num_block_layers[2], self.timestep, True, modified=True)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
        self.spike_func = MLF_unit(self.timestep)

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        # expand x to 4 time-steps
        b_size = x.shape[0]
        x_temp = torch.zeros((self.timestep * b_size,) + x.shape[1:], device=x.device)
        for t in range(self.timestep):
            x_temp[t*b_size:(t+1)*b_size, ...] = x

        out = self.spike_func(self.bn1(self.conv1(x_temp)))

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.avg_pool(out)
        out = out.view(out.shape[0], -1)

        out = self.fc(out)
        bs = int(out.shape[0] / self.timestep)
        o = torch.zeros((bs,) + out.shape[1:], device=out.device)
        for t in range(self.timestep):
            o += out[t*bs:(t+1)*bs, ...]
        o /= self.timestep
        return o


def resnet14(timestep):
    return ResNet(BasicBlock, [2, 2, 2], timestep)

def resnet20(timestep):
    return ResNet(BasicBlock, [3, 3, 3], timestep)

def resnet32(timestep):
    return ResNet(BasicBlock, [5, 5, 5], timestep)

def resnet44(timestep):
    return ResNet(BasicBlock, [7, 7, 7], timestep)

def resnet68(timestep):
    return ResNet(BasicBlock, [11, 11, 11], timestep)

def get_resnet_spiking(name, timestep):
    if name == "resnet14":
        return resnet14(timestep)
    elif name == "resnet20":
        return resnet20(timestep)
    elif name == "resnet32":
        return resnet32(timestep)
    elif name == "resnet44":
        return resnet44(timestep)
    elif name == "resnet68":
        return resnet68(timestep)