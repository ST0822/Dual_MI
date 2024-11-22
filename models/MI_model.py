import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch


class MIFCNet(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(MIFCNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, latent_dim)
        self.fc_output = nn.Linear(latent_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.sigmoid(self.fc_output(x))
        return x

class Mine(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 1)
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.normal_(self.fc2.weight, std=0.02)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.normal_(self.fc3.weight, std=0.02)
        nn.init.constant_(self.fc3.bias, 0)

    def forward(self, input):
        output = F.elu(self.fc1(input))
        output = F.elu(self.fc2(output))
        output = self.fc3(output)
        return output


class MIFCNetDIM(nn.Module):
    """Simple custom network for computing MI.

    """
    def __init__(self, n_input, n_units, bn=False):
        """
        Args:
            n_input: Number of input units.
            n_units: Number of output units.
        """
        super().__init__()

        self.bn = bn

        assert(n_units >= n_input)

        self.linear_shortcut = nn.Linear(n_input, n_units)
        self.block_nonlinear = nn.Sequential(
            nn.Linear(n_input, n_units, bias=False),
            nn.BatchNorm1d(n_units),
            nn.ReLU(),
            nn.Linear(n_units, n_units)
        )

        # initialize the initial projection to a sort of noisy copy
        eye_mask = np.zeros((n_units, n_input), dtype=np.uint8)
        for i in range(n_input):
            eye_mask[i, i] = 1

        self.linear_shortcut.weight.data.uniform_(-0.01, 0.01)
        self.linear_shortcut.weight.data.masked_fill_(torch.tensor(eye_mask), 1.)
        self.block_ln = nn.LayerNorm(n_units)

    def forward(self, x):
        h = self.block_nonlinear(x) + self.linear_shortcut(x)

        if self.bn:
            h = self.block_ln(h)
        return h