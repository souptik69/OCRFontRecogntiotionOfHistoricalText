import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from utils.ConvolutionUtils import compute_unpadded_width_after_conv


class GCRNNEncoder(pl.LightningModule):
    def __init__(self, n_channels=1, stride_conv1 = 2):
        # TODO: add parameters
        super(GCRNNEncoder, self).__init__()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.conv1 = nn.Conv2d(n_channels, 8, (3, 3), stride=stride_conv1)
        self.conv2 = nn.Conv2d(8, 16, (2, 4))
        self.conv_gate1 = nn.Conv2d(16, 16, (3, 3))
        self.conv3 = nn.Conv2d(16, 32, (3, 3))
        self.conv_gate2 = nn.Conv2d(32, 32, (3, 3))
        self.conv4 = nn.Conv2d(32, 64, (2, 4))
        self.conv_gate3 = nn.Conv2d(64, 64, (3, 3))
        self.conv5 = nn.Conv2d(64, 128, (3, 3))
        self.lstm1 = torch.nn.LSTM(input_size=128, batch_first=True, hidden_size=128, bidirectional=True)
        self.lstm2 = torch.nn.LSTM(input_size=128, batch_first=True, hidden_size=128, bidirectional=True)
        self.linear1 = torch.nn.Linear(128, 128)
        self.padding_params = [[3, stride_conv1, 0], [4, 1, 0], [3, 1, 0], [4, 1, 0], [3, 1, 0]]

    def forward(self, x, padding=None):
        # if x.shape[-1] < 23:
        #     helper = torch.zeros(*x.shape[:-1],23)
        #     helper[:,:,:,:x.shape[-1]] = x
        #     x = helper
        x = self.tanh(self.conv1(x))
        x = self.tanh(self.conv2(x))
        x = self.sigmoid(self.conv_gate1(F.pad(x, (1, 1, 1, 1)))) * x
        x = self.tanh(self.conv3(x))
        x = self.sigmoid(self.conv_gate2(F.pad(x, (1, 1, 1, 1)))) * x
        x = self.tanh(self.conv4(x))
        x = self.sigmoid(self.conv_gate3(F.pad(x, (1, 1, 1, 1)))) * x
        x = self.tanh(self.conv5(x))
        x, _ = torch.max(x, -2)
        x = x.permute(0, 2, 1)
        if padding != None:
            unpadded_widths = compute_unpadded_width_after_conv(padding, self.padding_params)
            x = pack_padded_sequence(x, unpadded_widths, batch_first=True, enforce_sorted=False)
        x, (h1, c1) = self.lstm1(x)
        if padding != None:
            x, _ = pad_packed_sequence(x, batch_first=True)
        x = x[:, :, :int(int(x.shape[2]) / 2)] + x[:, :, int(int(x.shape[2]) / 2):]
        x = self.tanh(self.linear1(x))
        if padding != None:
            unpadded_widths = compute_unpadded_width_after_conv(padding, self.padding_params)
            x = pack_padded_sequence(x, unpadded_widths, batch_first=True, enforce_sorted=False)
        x, (h2, c2) = self.lstm2(x)
        if padding != None:
            x, _ = pad_packed_sequence(x, batch_first=True)
        x = x[:, :, :int(int(x.shape[2]) / 2)] + x[:, :, int(int(x.shape[2]) / 2):]
        return x

class GCRNNDecoder(pl.LightningModule):
    def __init__(self, vocab_size, n_features):
        super(GCRNNDecoder, self).__init__()
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(n_features, vocab_size),
        )

    def forward(self,x):
        x = self.classifier(x)
        return F.log_softmax(x, dim=2)


class GCRNN(pl.LightningModule):
    def __init__(self, vocab_size, n_features=128):
        # TODO: add params!
        super(GCRNN, self).__init__()
        self.encoder = GCRNNEncoder()
        self.decoder = GCRNNDecoder(n_features=n_features, vocab_size=vocab_size)

    def forward(self, x, padding=None):
        return self.decoder(self.encoder(x, padding))


if __name__ == "__main__":
    enc = GCRNNEncoder()
    x = torch.randn(1,1,64,22)
    enc(x)