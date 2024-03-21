import numpy as np
import torch
from torch import nn


class LinRecClassifierCnn(nn.Module):

        """An 1D Convulational Neural Network for Sentence Classification."""
        def __init__(self,
                     content_length=102,
                     embed_dim=300,
                     filter_sizes=[3, 4, 5],
                     num_filters=[100, 100, 100],
                     num_classes=1,
                     dropout=0.5):

            super(LinRecClassifierCnn, self).__init__()
            # Embedding layer

            self.embed_dim = embed_dim
            self.embedding = nn.Embedding(num_embeddings=content_length,
                                          embedding_dim=self.embed_dim,
                                          padding_idx=1)

            # Conv Network
            self.conv1d_list = nn.ModuleList([
                nn.Conv1d(in_channels=self.embed_dim,
                          out_channels=num_filters[i],
                          kernel_size=filter_sizes[i])
                for i in range(len(filter_sizes))
            ])
            # Fully-connected layer and Dropout
            self.fc = nn.Linear(np.sum(num_filters), num_classes)
            self.dropout = nn.Dropout(p=dropout)

        def forward(self, input_ids, padding=None):
            """Perform a forward pass through the network.

            Args:
                input_ids (torch.Tensor): A tensor of token ids with shape
                    (batch_size, max_sent_length)

            Returns:
                logits (torch.Tensor): Output logits with shape (batch_size,
                    n_classes)
            """

            # Get embeddings from `input_ids`. Output shape: (b, max_len, embed_dim)
            x_embed = self.embedding(input_ids.long()).float()

            # Permute `x_embed` to match input shape requirement of `nn.Conv1d`.
            # Output shape: (b, embed_dim, max_len)
            x_reshaped = x_embed.permute(0, 2, 1)

            # Apply CNN and ReLU. Output shape: (b, num_filters[i], L_out)
            x_conv_list = [torch.relu(conv1d(x_reshaped)) for conv1d in self.conv1d_list]

            # Max pooling. Output shape: (b, num_filters[i], 1)
            x_pool_list = [torch.max_pool1d(x_conv, kernel_size=x_conv.shape[2])
                           for x_conv in x_conv_list]

            # Concatenate x_pool_list to feed the fully connected layer.
            # Output shape: (b, sum(num_filters))
            x_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list],
                             dim=1)

            # Compute logits. Output shape: (b, n_classes)
            logits = self.fc(self.dropout(x_fc))

            return logits.squeeze()
