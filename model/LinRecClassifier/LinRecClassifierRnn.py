import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LinRecClassifierRnn(nn.Module):

    def __init__(self, max_content_length=102, hidden_dimension=256):
        super(LinRecClassifierRnn, self).__init__()
        self.emb = nn.Embedding(max_content_length, hidden_dimension)
        self.blstm = nn.LSTM(input_size=hidden_dimension,
                             hidden_size=hidden_dimension,
                             num_layers=2,
                             batch_first=True,
                             bidirectional=True,
                             dropout=0.2)

        self.embedding_collector = nn.Sequential(
            nn.Linear(hidden_dimension, int(hidden_dimension/2)),
            nn.Linear(int(hidden_dimension/2), 1)
        )

        self.classifier = nn.Linear(max_content_length, 1)

    def forward(self, x):
        emb = self.emb(x)
        blstm_output, _ = self.blstm(emb)

        summed_output = blstm_output[:, :, :int(blstm_output.shape[2] / 2)] + blstm_output[:, :, int(blstm_output.shape[2] / 2):]
        collected_output = self.embedding_collector(summed_output)
        collected_output = torch.squeeze(collected_output)
        output = self.classifier(collected_output)
        return torch.squeeze(output)


if __name__ == "__main__":
    m = LinRecClassifierRnn()

    x = abs(torch.randn(5, 102).long())
    out = m(x)
    print(out.shape)
