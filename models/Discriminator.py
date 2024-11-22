import torch
from torch import nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_discriminator(inputs, Discriminator, Generator, opti_loss, lamda):
    crition = nn.BCELoss()
    labelt_inp = torch.ones((inputs.shape[0], 1)).to(device)
    labelt_out = torch.zeros((inputs.shape[0], 1)).to(device)

    Generator.eval()
    opti_loss.zero_grad()
    with torch.no_grad():
        latent_code, feature_map, feature_map_list = Generator(inputs)
        recon = Generator.decoder(latent_code)

    label_inp = Discriminator(inputs)
    label_out = Discriminator(recon)
    loss_Dis = lamda * crition(torch.cat((label_inp, label_out), 0),
                               torch.cat((labelt_inp, labelt_out), 0))
    loss_Dis.backward()
    opti_loss.step()
    return loss_Dis


class Discriminator_Channel_62(nn.Module):
    def __init__(self, n_layer, n_filters, input_size):
        super(Discriminator_Channel_62, self).__init__()
        self.conv1 = nn.Conv2d(1, int(8 * n_filters), (1, int(input_size / 2 + 1)), stride=1,
                               padding=(0, int(input_size / 4)))
        self.batchNorm1 = nn.BatchNorm2d(8 * n_filters, False)
        self.length = input_size / 32
        # Layer 2：spatial convolution for channel dim
        self.depthwiseconv2 = nn.Conv2d(int(8 * n_filters), int(16 * n_filters), (62, 1), padding=0)
        self.batchNorm2 = nn.BatchNorm2d(int(16 * n_filters), False)
        self.pooling1 = nn.MaxPool2d((1, 4), return_indices=False)
        self.separa1conv3 = nn.Conv2d(int(16 * n_filters), int(16 * n_filters), (1, int(input_size / 8 + 1)), stride=1,
                                      padding=(0, int(input_size / 16)), groups=int(16 * n_filters))
        self.separa2conv4 = nn.Conv2d(int(16 * n_filters), int(8 * n_filters), 1)
        self.batchNorm3 = nn.BatchNorm2d(int(8 * n_filters), False)
        self.pooling2 = nn.MaxPool2d((1, 8), return_indices=False)
        self.fc1 = nn.Linear(int(self.length * 8), 1)

    def forward(self, x):
        # encoder
        x = self.conv1(x)
        x = self.batchNorm1(x)
        # Layer 2
        x = self.depthwiseconv2(x)
        x = self.batchNorm2(x)
        x = F.elu(x)
        x = self.pooling1(x)
        # Layer 3
        x = self.separa1conv3(x)
        x = self.separa2conv4(x)
        x = self.batchNorm3(x)
        x = F.elu(x)
        x = self.pooling2(x)

        # Layer 4：FC Layer
        x = x.reshape((x.shape[0], int(self.length * 8)))
        x = self.fc1(x)
        x = torch.sigmoid(x)
        return x


class Discriminator_Channel_32(nn.Module):
    def __init__(self, n_layer, n_filters, input_size):
        super(Discriminator_Channel_32, self).__init__()
        self.conv1 = nn.Conv2d(1, int(8 * n_filters), (1, int(input_size / 2 + 1)), stride=1,
                               padding=(0, int(input_size / 4)))  # (16, C, T)
        self.batchNorm1 = nn.BatchNorm2d(8 * n_filters, False)
        self.length = input_size / 32
        # Layer 2：spatial convolution for channel dim
        self.depthwiseconv2 = nn.Conv2d(int(8 * n_filters), int(16 * n_filters), (32, 1), padding=0)
        self.batchNorm2 = nn.BatchNorm2d(int(16 * n_filters), False)
        self.pooling1 = nn.MaxPool2d((1, 4), return_indices=False)
        self.separa1conv3 = nn.Conv2d(int(16 * n_filters), int(16 * n_filters), (1, int(input_size / 8 + 1)), stride=1,
                                      padding=(0, int(input_size / 16)), groups=int(16 * n_filters))
        self.separa2conv4 = nn.Conv2d(int(16 * n_filters), int(8 * n_filters), 1)
        self.batchNorm3 = nn.BatchNorm2d(int(8 * n_filters), False)
        self.pooling2 = nn.MaxPool2d((1, 8), return_indices=False)
        self.fc1 = nn.Linear(int(self.length * 8), 1)

    def forward(self, x):
        # encoder
        x = self.conv1(x)
        x = self.batchNorm1(x)
        # Layer 2
        x = self.depthwiseconv2(x)
        x = self.batchNorm2(x)
        x = F.elu(x)
        x = self.pooling1(x)  # get data and their index after pooling
        # Layer 3
        x = self.separa1conv3(x)
        x = self.separa2conv4(x)
        x = self.batchNorm3(x)
        x = F.elu(x)
        x = self.pooling2(x)

        # Layer 4：FC Layer
        x = x.reshape((x.shape[0], int(self.length * 8)))
        x = self.fc1(x)
        x = torch.sigmoid(x)
        return x