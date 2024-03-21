from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CyclicLR, ReduceLROnPlateau, CosineAnnealingLR

from model.unet.AttentionUnet import AttentionUnet
from model.unet.Unet import Unet
from utils.metrics.UnetMetricUtils import compute_pixel_accuracy, compute_dice_coefficient
from utils.constants import *
from utils.type.LrScheduleType import LrScheduleType


class UnetModel(pl.LightningModule):
    def __init__(self, lr_schedule_type: LrScheduleType = LrScheduleType.NONE, architecture="unet", balance=None, img_chan = 1):
        super().__init__()
        if architecture == "unet":
            self.unet = Unet(in_ch=img_chan)
        elif architecture == "att_unet":
            self.unet = AttentionUnet(img_ch=img_chan)
        # if len(balance)==0:
        #     balance = torch.Tensor([1])
        self.loss = nn.BCEWithLogitsLoss(pos_weight=balance)

        self.lr_scheduler = lr_schedule_type
        self.default_lr = 1e-3
        self.lr = None

    def forward(self, x):
        output = self.unet(x)
        return output

    def training_step(self, train_batch, batch_idx):
        loss = self._shared_step(train_batch, "train")
        return loss

    def validation_step(self, val_batch, batch_idx):
        loss = self._shared_step(val_batch, "val")
        return loss

    def test_step(self, test_batch, test_idx):
        loss = self._shared_step(test_batch, "test")
        return loss

    def _shared_step(self, batch, step: str):
        image, mask = batch[IMAGE], batch[MASK]
        logits = self.forward(image)
        loss = self.loss(logits, mask)
        self.log(f"{step}_loss", loss)

        if step == "test":
            sig = torch.sigmoid(logits)
            preds = sig > 0.5
            pixel_acc = compute_pixel_accuracy(preds, mask)
            dice_loss = compute_dice_coefficient(preds, mask)
            self.log("point_acc", pixel_acc)
            self.log("dice_loss", dice_loss)

        return loss

    def configure_optimizers(self):
        # TODO outsource
        optimizer = torch.optim.AdamW(self.parameters(), lr=(self.lr or self.default_lr))

        output = {
            'optimizer': optimizer
        }

        if self.lr_scheduler == LrScheduleType.NONE:
            return output

        elif self.lr_scheduler == LrScheduleType.REDUCE_ON_PLATEAU:
            output["scheduler"] = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
            output["monitor"] = "val_loss"

        elif self.lr_scheduler == LrScheduleType.COSINE_ANNEALING:
            output["scheduler"] = CosineAnnealingLR(optimizer, T_max=10)

        elif self.lr_scheduler == LrScheduleType.CYCLIC:
            # cycle_momentum has to be set to false in order to work with adam
            output["scheduler"] = CyclicLR(optimizer, base_lr=((self.lr or self.default_lr) / 10),
                                           max_lr=(self.lr or self.default_lr), step_size_up=80, cycle_momentum=False)

        return output

    def _forward_unimplemented(self, *input: Any) -> None:
        pass
