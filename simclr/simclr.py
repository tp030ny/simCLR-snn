import torch.nn as nn
import torchvision

from simclr.modules.resnet_hacks import modify_resnet_model
from simclr.modules.identity import Identity
from simclr.modules.spike_layer import *

class SimCLR(nn.Module):
    """
    We opt for simplicity and adopt the commonly used ResNet (He et al., 2016) to obtain hi = f(x ̃i) = ResNet(x ̃i) where hi ∈ Rd is the output after the average pooling layer.
    SNN-version
    """

    def __init__(self, encoder, projection_dim, n_features, timestep):
        super(SimCLR, self).__init__()

        self.timestep = timestep
        self.encoder = encoder
        self.n_features = n_features

        # Replace the fc layer with an Identity function
        self.encoder.fc = Identity()

        # We use a MLP with one hidden layer to obtain z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
        self.projector = nn.Sequential(
            nn.Linear(self.n_features, self.n_features, bias=False),
            # nn.ReLU(),
            MLF_unit(self.timestep),
            nn.Linear(self.n_features, projection_dim, bias=False),
        )

    def forward(self, x_i, x_j):
        h_i_temp = self.encoder(x_i)
        h_j_temp = self.encoder(x_j)

        timestep = self.timestep
        b_size = h_i_temp.shape[0]
        h_i = torch.zeros((timestep * b_size,) + h_i_temp.shape[1:], device=h_i_temp.device)
        h_j = torch.zeros((timestep * b_size,) + h_j_temp.shape[1:], device=h_j_temp.device)
        for t in range(timestep):
            h_i[t*b_size:(t+1)*b_size, ...] = h_i_temp
            h_j[t*b_size:(t+1)*b_size, ...] = h_j_temp

        z_i_temp = self.projector(h_i)
        z_j_temp = self.projector(h_j)

        z_i = torch.zeros((b_size,) + z_i_temp.shape[1:], device=z_i_temp.device)
        z_j = torch.zeros((b_size,) + z_j_temp.shape[1:], device=z_j_temp.device)
        for t in range(self.timestep):
            z_i += z_i_temp[t*b_size:(t+1)*b_size, ...]
            z_j += z_j_temp[t*b_size:(t+1)*b_size, ...]
        z_i /= self.timestep
        z_j /= self.timestep
        return h_i, h_j, z_i, z_j
