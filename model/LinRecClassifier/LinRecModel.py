import pytorch_lightning as pl

import torch
from torch import nn

from model.LinRecClassifier.LinRecClassifierCnn import LinRecClassifierCnn
from model.LinRecClassifier.LinRecClassifierRnn import LinRecClassifierRnn
from model.htr.ctc_model import CTCModule

from utils.constants import *
from sklearn.metrics import accuracy_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from utils.metrics.TextConverter import ctc_remove_successives_identical_ind_batch
import pandas as pd


class LinRecModel(pl.LightningModule):

    def __init__(self, htr_checkpoint_path=None, loss_balancing=None, architecture="rnn",
                 max_content_len=102):
        super(LinRecModel, self).__init__()
        self.architecture = architecture
        self.max_content_len = max_content_len

        # load htr model
        self.htr = CTCModule.load_from_checkpoint(htr_checkpoint_path)
        self.htr.freeze()

        if architecture == "cnn":
            self.model = LinRecClassifierCnn(content_length=max_content_len)
        elif architecture == "rnn":
            self.model = LinRecClassifierRnn(max_content_length=max_content_len, hidden_dimension=64)

        # get loss function
        self.loss = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=loss_balancing)
        self.sigmoid = nn.Sigmoid()

        self.default_lr = 1e-3
        self.lr = None

    def forward(self, x, padding=None):
        output = self.htr(x, padding)

        aggregated_output = torch.max(output, dim=2)[1]
        pad_value = self.htr.alphabet.toPosition["<pad>"]
        blank_value = self.htr.alphabet.toPosition["<blank>"]
        cleared_output = ctc_remove_successives_identical_ind_batch(aggregated_output, remove_ind=[blank_value], pad_ind=pad_value)

        padded_output = [torch.nn.functional.pad(x, pad=(0, self.max_content_len - x.numel()),
                                                 mode='constant', value=pad_value) for x in cleared_output]
        class_input = torch.stack(padded_output)

        out = self.model(class_input.type(torch.LongTensor).cuda())
        return cleared_output, out

    def training_step(self, training_batch, batch_idx):
        loss, _, _ = self._shared_step(training_batch, "train")
        return loss

    def validation_step(self, val_batch, batch_idx):
        loss, _, _ = self._shared_step(val_batch, "val")
        return loss

    def test_step(self, test_batch, batch_idx):
        loss, preds, targets = self._shared_step(test_batch, "test")
        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs) -> None:
        preds = torch.cat([tmp['preds'] for tmp in outputs])
        targets = torch.cat([tmp['targets'] for tmp in outputs])

        cm = confusion_matrix(targets.cpu(), preds.cpu()).astype(int)
        df_cm = pd.DataFrame(cm, index=range(2), columns=range(2))
        plt.figure(figsize=(10, 7))
        fig_ = sns.heatmap(df_cm, annot=True, cmap='Spectral', fmt="d").get_figure()
        plt.close(fig_)
        self.logger.experiment.add_figure("Confusion matrix", fig_, self.current_epoch)

    def _shared_step(self, batch, step: str):
        x, y, padding = batch[IMAGE], batch[RECIPIENT], batch[UNPADDED_IMAGE_WIDTH]

        t, logits = self.forward(x, padding)
        loss = self.loss(logits, y)
        pred = None

        if step != "train":
            targets = y.cpu()
            pred = (self.sigmoid(logits) > 0.5).cpu()
            self.log(f"{step}_acc", accuracy_score(pred, targets))
            self.log(f"{step}_f1", f1_score(pred, targets))

        self.log(f"{step}_loss", loss)
        return loss, pred, y

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=(self.lr or self.default_lr))
        return optimizer
