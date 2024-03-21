import torch
from torch import nn
from ctcdecode import CTCBeamDecoder

class LinRecConf(nn.Module):

    def __init__(self, dimension=10, alphabet=None):
        super(LinRecConf, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, dimension),
            nn.ReLU(),
            nn.Linear(dimension, 1)
        )

        labels = [v for v in alphabet.keys()]
        labels[0] = "~"
        labels[1] = "$"
        self.decoder = CTCBeamDecoder(
            labels=labels,
            log_probs_input=True
        )

    def forward(self, x, padding=None):
        # get confidence
        beam_results, beam_scores, timesteps, out_seq_len = self.decoder.decode(x, padding)
        p = torch.true_divide(1, torch.exp(beam_scores[:, 0]))
        # fc
        out = self.fc(p.unsqueeze(1))
        return out.squeeze(1)
