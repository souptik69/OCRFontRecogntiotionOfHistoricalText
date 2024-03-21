import os

import torch
from torch import nn
import pytorch_lightning as pl

from model.htr.gcrnn import GCRNN
from utils.alphabet import Alphabet

# from htr_toolbox.model.ctc import get_model
# from htr_toolbox.data.IAMDataModule import IAMDataModule
# from htr_toolbox.data.IAMDataModuleTest import IAMDataModuleTest
# from htr_toolbox.data.NBBDataModule import NBBTrainModule, NBBTestModule
# from htr_toolbox.util.constants import *
# from htr_toolbox.model.util import compute_unpadded_width_after_conv
# from htr_toolbox.data.util.alphabet import Alphabet
# from htr_toolbox.metric.metric import cer, wer
# from htr_toolbox.data.util.ctc import ctc_remove_successives_from_batch


class CTCModule(pl.LightningModule):
    def __init__(self, alphabet="NBB", model="gcrnn"):
        super(CTCModule, self).__init__()

        self.save_hyperparameters()

        self.alphabet = Alphabet(dataset=alphabet)
        if model == "gcrnn":
            self.model = GCRNN(vocab_size=len(self.alphabet.toCharacter))

        self.loss = nn.CTCLoss(zero_infinity=True, reduction='mean')
        self.lr = 0.0004

    # def training_step(self, batch, batch_idx):
    #     pred = self.forward(batch[IMAGE], padding=batch[UNPADDED_IMAGE_WIDTH])
    #     loss = self.loss(pred.permute(1,0,2), batch[TEXT_LOGITS],
    #                      compute_unpadded_width_after_conv(batch[UNPADDED_IMAGE_WIDTH], self.model.encoder.padding_params),
    #                      batch[UNPADDED_TEXT_LEN])
    #     self.log("train/loss", loss, on_epoch=True)
    #     return loss

    # def validation_step(self, batch, batch_idx):
    #     pred = self.forward(batch[IMAGE], padding=batch[UNPADDED_IMAGE_WIDTH])
    #     loss = self.loss(pred.permute(1,0,2), batch[TEXT_LOGITS],
    #                      compute_unpadded_width_after_conv(batch[UNPADDED_IMAGE_WIDTH], self.model.encoder.padding_params),
    #                      batch[UNPADDED_TEXT_LEN])
    #     pred_str = self.alphabet.batch_logits_to_string_list(ctc_remove_successives_from_batch(torch.argmax(pred, dim=2).long(), [0,1]))
    #     self.log("val/cer", cer(str_gt=batch[TEXT], str_pred=pred_str), on_epoch=True, on_step=True)
    #     self.log("val/wer", wer(str_gt=batch[TEXT], str_pred=pred_str, seperators=self.alphabet.seperators), on_epoch=True, on_step=True)
    #     if batch_idx == 0:
    #         for idx in range(len(batch[IMAGE])):
    #             image = wandb.Image(batch[IMAGE][idx], caption="GT:{} ||| PRED:{}".format(batch[TEXT][idx], pred_str[idx]))
    #             self.logger.experiment.log({"val example {}".format(idx): image})
    #             if idx>3:
    #                 break
    #     self.log("val/loss", loss, on_epoch=True, on_step=True)
    #     return loss
    #
    # def test_step(self, batch, batch_idx):
    #     pred = self.forward(batch[IMAGE], padding=batch[UNPADDED_IMAGE_WIDTH])
    #     loss = self.loss(pred.permute(1,0,2), batch[TEXT_LOGITS],
    #                      compute_unpadded_width_after_conv(batch[UNPADDED_IMAGE_WIDTH], self.model.encoder.padding_params),
    #                      batch[UNPADDED_TEXT_LEN])
    #     pred_str = self.alphabet.batch_logits_to_string_list(ctc_remove_successives_from_batch(torch.argmax(pred, dim=2).long(), [0,1]))
    #     self.log("test/cer", cer(str_gt=batch[TEXT], str_pred=pred_str), on_epoch=True, on_step=True)
    #     self.log("test/wer", wer(str_gt=batch[TEXT], str_pred=pred_str, seperators=self.alphabet.seperators), on_epoch=True, on_step=True)
    #     if batch_idx == 0:
    #         for idx in range(len(batch[IMAGE])):
    #             image = wandb.Image(batch[IMAGE][idx], caption="GT:{} ||| PRED:{}".format(batch[TEXT][idx], pred_str[idx]))
    #             self.logger.experiment.log({"test example {}".format(idx): image})
    #     self.log("test/loss", loss, on_epoch=True, on_step=True)
    #     return loss

    def forward(self,x, padding=None):
        return self.model(x, padding)

    def configure_optimizers(self):
        optimizer = torch.optim.RMSprop(self.parameters(), lr=self.lr)
        return optimizer


if __name__ == "__main__":
    system = CTCModule(alphabet="NBB").load_from_checkpoint("../../wp3/htr/htr_ctc.ckpt")
    # Model was trained on binarized images with fixed height 64. (Text = white; Background = black)
    t = torch.randn(10,1,64,2300)
    print(system(t).shape)