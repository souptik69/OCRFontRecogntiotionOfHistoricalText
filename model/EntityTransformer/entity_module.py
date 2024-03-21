from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
import wandb
import torch
import pytorch_lightning as pl

from model.EntityTransformer.entity_model import EntityModel
from model.EntityTransformer.entity_model import EntityModelFont
from utils.alphabet import Alphabet
from utils.constants import *
from utils.SmoothCE import SmoothCE
from utils.metrics.TextMetricUtils import cer, wer, font_acc, int_cer
from utils.noisy_teacher_forcing import NoisyTeacherForcing
from utils.noisy_teacher_forcing import NoisyTeacherFont

class EntityModule(pl.LightningModule):

    def __init__(self, vocab_size, eps=0., lr=1e-3, hidden_size=512, n_head=1, dropout=0.1, noise_teacher_forcing=0.2):
        super(EntityModule, self).__init__()

        self.save_hyperparameters()
        self.A = Alphabet(dataset="NBB", mode="s2s_recipient")
        self.noisy_teacher = NoisyTeacherForcing(A_size=len(self.A.toPosition),noise_prob=noise_teacher_forcing)
        self.lr = lr

        self.model = EntityModel(vocab_size=vocab_size, hidden_size=hidden_size, n_head=n_head,
                                 dropout=dropout)
        self.criterion = SmoothCE(eps=eps, reduction="mean", trg_pad_idx=self.A.toPosition[PAD])

    def forward(self, x, tgt, tgt_mask=None, tgt_key_padding_mask=None):
        return self.model(x=x, tgt=tgt, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)

    def training_step(self, batch, batch_idx):
        x, tgt, tgt_key_padding_mask, tgt_mask = batch[LINE_IMAGE], batch[S2S_TEXT], batch[TGT_KEY_PADDING_MASK], batch[TGT_MASK]
        # print(tgt.shape, tgt_mask.shape, tgt_key_padding_mask.shape)
        pred, attention = self.forward(x=x, tgt=self.noisy_teacher(tgt[:,:-1]), tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask[:,:-1])
        loss = self.criterion(pred, tgt[:,1:])
        self.log("train/loss",loss)
        end_tokens = [self.A.toPosition[END_OF_SEQUENCE_BODY], self.A.toPosition[END_OF_SEQUENCE_RECIPIENT]]
        pred_str, pred_recipient = self.A.batch_logits_to_string_list(torch.argmax(pred, dim=2).long().cpu(),
                                                                      stopping_logits=end_tokens)
        gt_str, gt_recipient = self.A.batch_logits_to_string_list(tgt[:, 1:].cpu(),
                                                                  stopping_logits=end_tokens)
        self.log("train/cer", cer(str_gt=gt_str, str_pred=pred_str), on_epoch=True, on_step=True)
        self.log("train/wer", wer(str_gt=gt_str, str_pred=pred_str, seperators=self.A.seperators),
                 on_epoch=True, on_step=True)
        acc = torch.true_divide(torch.sum(pred_recipient == gt_recipient), len(pred_recipient))
        self.log("train/recipient", acc, on_epoch=True, on_step=True)
        if batch_idx == 0:
            for idx in range(len(batch[LINE_IMAGE])):
                if idx > 2:
                    break
                image = wandb.Image(batch[LINE_IMAGE][idx],
                                    caption="GT:{} ||| PRED:{} ||| Recipient: GT:{} PRED:{}".format(
                                        gt_str[idx],
                                        pred_str[idx],
                                        self.A.toCharacter[int(gt_recipient[idx])],
                                        self.A.toCharacter[int(pred_recipient[idx])],
                                        # torch.argmax(pred, dim=2).long().cpu()[idx],
                                    ))
                self.logger.experiment.log({"train example {}".format(idx): image})
                att_img = attention[idx]
                att_img = att_img.reshape(att_img.shape[0], 2, -1)  # TODO: check 4 dynamically
                att_img = att_img.sum(dim=1)
                att_img -= att_img.min(1, keepdim=True)[0]
                att_img /= att_img.max(1, keepdim=True)[0]
                att_img_wandb = wandb.Image(att_img[:(~tgt_key_padding_mask[idx]).sum(), :].unsqueeze(0))
                self.logger.experiment.log({"train attention example {}".format(idx): att_img_wandb})
        return loss

    def validation_step(self, batch, batch_idx):
        x, tgt, tgt_key_padding_mask, tgt_mask = batch[LINE_IMAGE], batch[S2S_TEXT], batch[TGT_KEY_PADDING_MASK], batch[
            TGT_MASK]
        # print(tgt.shape, tgt_mask.shape, tgt_key_padding_mask.shape)

        # pertubated validation
        pred, attention = self.forward(x=x, tgt=self.noisy_teacher(tgt[:,:-1]), tgt_mask=tgt_mask,
                                       tgt_key_padding_mask=tgt_key_padding_mask[:, :-1])
        loss = self.criterion(pred, tgt[:, 1:])
        self.log("val_pert/loss", loss)
        end_tokens = [self.A.toPosition[END_OF_SEQUENCE_BODY], self.A.toPosition[END_OF_SEQUENCE_RECIPIENT]]
        pred_str, pred_recipient = self.A.batch_logits_to_string_list(torch.argmax(pred, dim=2).long().cpu(),
                                                                      stopping_logits=end_tokens)
        gt_str, gt_recipient = self.A.batch_logits_to_string_list(tgt[:, 1:].cpu(),
                                                                  stopping_logits=end_tokens)
        self.log("val_pert/cer", cer(str_gt=gt_str, str_pred=pred_str))
        self.log("val_pert/wer", wer(str_gt=gt_str, str_pred=pred_str, seperators=self.A.seperators))
        acc = torch.true_divide(torch.sum(pred_recipient == gt_recipient), len(pred_recipient))
        self.log("val_pert/recipient", acc)

        pred, attention = self.forward(x=x, tgt=tgt[:, :-1], tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask[:, :-1])
        loss = self.criterion(pred, tgt[:, 1:])
        self.log("val/loss",loss)
        end_tokens = [self.A.toPosition[END_OF_SEQUENCE_BODY], self.A.toPosition[END_OF_SEQUENCE_RECIPIENT]]
        pred_str, pred_recipient = self.A.batch_logits_to_string_list(torch.argmax(pred, dim=2).long().cpu(),
                                                      stopping_logits=end_tokens)
        gt_str, gt_recipient = self.A.batch_logits_to_string_list(tgt[:, 1:].cpu(),
                                                    stopping_logits=end_tokens)
        self.log("val/cer", cer(str_gt=gt_str, str_pred=pred_str))
        self.log("val/wer", wer(str_gt=gt_str, str_pred=pred_str, seperators=self.A.seperators))
        acc = torch.true_divide(torch.sum(pred_recipient==gt_recipient),len(pred_recipient))
        self.log("val/recipient",acc)
        if batch_idx == 0:
            for idx in range(len(batch[LINE_IMAGE])):
                if idx>2:
                    break
                image = wandb.Image(batch[LINE_IMAGE][idx],
                                    caption="GT:{} ||| PRED:{} ||| Recipient: GT:{} PRED:{}".format(
                                        gt_str[idx],
                                        pred_str[idx],
                                        self.A.toCharacter[int(gt_recipient[idx])],
                                        self.A.toCharacter[int(pred_recipient[idx])],
                                        # torch.argmax(pred, dim=2).long().cpu()[idx],
                                    ))
                self.logger.experiment.log({"val example {}".format(idx): image})
                att_img = attention[idx]
                att_img = att_img.reshape(att_img.shape[0], 2, -1)  # TODO: find 4 dynamically
                att_img = att_img.sum(dim=1)
                att_img -= att_img.min(1, keepdim=True)[0]
                att_img /= att_img.max(1, keepdim=True)[0]
                att_img_wandb = wandb.Image(att_img[:(~tgt_key_padding_mask[idx]).sum(),:].unsqueeze(0))
                self.logger.experiment.log({"val attention example {}".format(idx): att_img_wandb})

        return loss

    def test_step(self, batch, batch_idx):
        x, tgt, tgt_key_padding_mask, tgt_mask = batch[LINE_IMAGE], batch[S2S_TEXT], batch[TGT_KEY_PADDING_MASK], batch[
            TGT_MASK]
        # fair
        pred_logits = (torch.ones(size=tgt.shape)*self.A.toPosition[PAD]).long()
        pred = torch.ones(size=(*tgt.shape,len(self.A.toPosition)))
        pred_logits[:,0] = (torch.ones(size=pred_logits[:,0].shape)*self.A.toPosition[START_OF_SEQUENCE]).long()
        if x.is_cuda:
            pred = pred.cuda()
            pred_logits = pred_logits.cuda()
        for i in range(1,tgt.shape[-1]):
            out, attention = self.forward(x, tgt=pred_logits[:,:i])
            pred_logits[:,i] = torch.argmax(out, dim=2).long()[:,-1]
            pred[:,i-1] = out[:,-1,:]
        loss = self.criterion(pred[:,:-1], tgt[:, 1:])
        self.log("test/loss", loss)
        end_tokens = [self.A.toPosition[END_OF_SEQUENCE_BODY], self.A.toPosition[END_OF_SEQUENCE_RECIPIENT]]
        pred_str, pred_recipient = self.A.batch_logits_to_string_list(torch.argmax(pred, dim=2).long().cpu(),
                                                                      stopping_logits=end_tokens)
        gt_str, gt_recipient = self.A.batch_logits_to_string_list(tgt[:, 1:].cpu(),
                                                                  stopping_logits=end_tokens)
        self.log("test/cer", cer(str_gt=gt_str, str_pred=pred_str), on_epoch=True, on_step=True)
        self.log("test/wer", wer(str_gt=gt_str, str_pred=pred_str, seperators=self.A.seperators),
                 on_epoch=True, on_step=True)
        acc = torch.true_divide(torch.sum(pred_recipient == gt_recipient), len(pred_recipient))
        self.log("test/recipient", acc, on_epoch=True, on_step=True)

        # unfair
        pred, attention = self.forward(x=x, tgt=tgt[:, :-1], tgt_mask=tgt_mask,
                                       tgt_key_padding_mask=tgt_key_padding_mask[:, :-1])
        loss = self.criterion(pred, tgt[:, 1:])
        self.log("test_unfair/loss", loss)
        end_tokens = [self.A.toPosition[END_OF_SEQUENCE_BODY], self.A.toPosition[END_OF_SEQUENCE_RECIPIENT]]
        pred_str, pred_recipient = self.A.batch_logits_to_string_list(torch.argmax(pred, dim=2).long().cpu(),
                                                                      stopping_logits=end_tokens)
        gt_str, gt_recipient = self.A.batch_logits_to_string_list(tgt[:, 1:].cpu(),
                                                                  stopping_logits=end_tokens)
        self.log("test_unfair/cer", cer(str_gt=gt_str, str_pred=pred_str), on_epoch=True, on_step=True)
        self.log("test_unfair/wer", wer(str_gt=gt_str, str_pred=pred_str, seperators=self.A.seperators),
                 on_epoch=True, on_step=True)
        acc = torch.true_divide(torch.sum(pred_recipient == gt_recipient), len(pred_recipient))
        self.log("test_unfair/recipient", acc, on_epoch=True, on_step=True)


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
    
### Font detection ##


class EntityModuleFont(pl.LightningModule):

    def __init__(self, vocab_size,font_classes, eps=0., lr=1e-3, hidden_size=512, n_head=1, dropout=0.1, noise_teacher_forcing=0.2):
        super(EntityModuleFont, self).__init__()
        self.save_hyperparameters()
        self.font_classes = font_classes
        self.A = Alphabet(dataset="NBB", mode="attention")
        self.noisy_teacher = NoisyTeacherFont(character_vocab_size=len(self.A.toPosition),font_class_count=font_classes, noise_prob=noise_teacher_forcing)
        self.lr = lr

        self.model = EntityModelFont(vocab_size=vocab_size,font_classes=font_classes, hidden_size=hidden_size, n_head=n_head,
                                 dropout=dropout)
        self.criterion_char = SmoothCE(eps=eps, reduction="mean", trg_pad_idx=self.A.toPosition['<pad>'])
        self.criterion_font = SmoothCE(eps=eps, reduction="mean", trg_pad_idx=0)
    
    def forward(self, x, tgt, tgt_mask=None, tgt_key_padding_mask=None):
        return self.model(x=x, tgt=tgt, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
    
    def training_step(self, batch, batch_idx):
        x, tgt, tgt_key_padding_mask, tgt_mask = batch['LINE_IMAGE'], batch['S2S'], batch['TGT_KEY_PADDING_MASK'], batch['TGT_KEY_MASK']
        pred_char, pred_font, attention = self.forward(x=x, tgt=self.noisy_teacher(tgt[:,:-1]), tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask[:,:-1])
        loss_char = self.criterion_char(pred_char, tgt[:,1:,0])
        self.log("train/loss_char",loss_char,batch_size=tgt.shape[0],sync_dist=True)
        loss_font = self.criterion_font(pred_font, tgt[:,1:,1])
        self.log("train/loss_font", loss_font,batch_size=tgt.shape[0],sync_dist=True)
        weight_char = 1.0
        weight_font = 1.0
        total_loss = (weight_char * loss_char) + (weight_font * loss_font)
        self.log("train/total_loss", total_loss,batch_size=tgt.shape[0],sync_dist=True)
        end_tokens = [self.A.toPosition['<eos>']]
        pred_str = self.A.batch_logits_to_string_list_font(torch.argmax(pred_char, dim=2).long().cpu(),
                                                                      stopping_logits=end_tokens)
        gt_str = self.A.batch_logits_to_string_list_font(tgt[:, 1:,0].cpu(),stopping_logits=end_tokens)
        self.log("train/cer", cer(str_gt=gt_str, str_pred=pred_str), on_epoch=True, on_step=True,batch_size=tgt.shape[0],sync_dist=True)
        end_token = [13]
        pred_ft = self.A.batch_logits_to_list_font(torch.argmax(pred_font, dim=2).long().cpu(),
                                                                stopping_logits=end_token)
        gt_ft = self.A.batch_logits_to_list_font(tgt[:, 1:,1].cpu(),stopping_logits=end_token)
        self.log("train/font_acc", font_acc(font_gt=gt_ft, font_pred=pred_ft), on_epoch=True, on_step=True,batch_size=tgt.shape[0],sync_dist=True )

        if batch_idx == 0:
            for idx in range(len(batch['LINE_IMAGE'])):
                if idx > 2:
                    break
                image = wandb.Image(batch['LINE_IMAGE'][idx],
                                    caption="GT:{} ||| PRED:{} ||| FONT GT:{} PRED:{}".format(
                                        gt_str[idx],
                                        pred_str[idx],
                                        tgt[:,1:,1].cpu()[idx],
                                        torch.argmax(pred_font, dim=2).long().cpu()[idx],
                                        # torch.argmax(pred, dim=2).long().cpu()[idx],
                                    ))
                self.logger.experiment.log({"train example {}".format(idx): image})
                att_img = attention[idx]
                att_img = att_img.reshape(att_img.shape[0], 2, -1)  # TODO: check 4 dynamically
                att_img = att_img.sum(dim=1)
                att_img -= att_img.min(1, keepdim=True)[0]
                att_img /= att_img.max(1, keepdim=True)[0]
                att_img_wandb = wandb.Image(att_img[:(~tgt_key_padding_mask[idx]).sum(), :].unsqueeze(0))
                self.logger.experiment.log({"train attention example {}".format(idx): att_img_wandb})
        # torch.cuda.empty_cache()
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        x, tgt, tgt_key_padding_mask, tgt_mask = batch['LINE_IMAGE'], batch['S2S'], batch['TGT_KEY_PADDING_MASK'], batch['TGT_KEY_MASK']
        
        # pertubated validation
        pred_char, pred_font, attention = self.forward(x=x, tgt=self.noisy_teacher(tgt[:,:-1]), tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask[:,:-1])
        loss_char = self.criterion_char(pred_char, tgt[:,1:,0])
        self.log("val_pert/loss_char",loss_char,batch_size=tgt.shape[0],sync_dist=True)
        loss_font = self.criterion_font(pred_font, tgt[:,1:,1])
        self.log("val_pert/loss_font", loss_font,batch_size=tgt.shape[0],sync_dist=True)
        weight_char = 1.0
        weight_font = 1.0
        total_loss = (weight_char * loss_char) + (weight_font * loss_font)
        self.log("val_pert/total_loss", total_loss,batch_size=tgt.shape[0],sync_dist=True)
        end_tokens = [self.A.toPosition['<eos>']]
        pred_str = self.A.batch_logits_to_string_list_font(torch.argmax(pred_char, dim=2).long().cpu(),
                                                                      stopping_logits=end_tokens)
        gt_str = self.A.batch_logits_to_string_list_font(tgt[:, 1:,0].cpu(),stopping_logits=end_tokens)
        self.log("val_pert/cer", cer(str_gt=gt_str, str_pred=pred_str), on_epoch=True, on_step=True,batch_size=tgt.shape[0],sync_dist=True)

        end_token = [13]
        pred_ft = self.A.batch_logits_to_list_font(torch.argmax(pred_font, dim=2).long().cpu(),
                                                                stopping_logits=end_token)
        gt_ft = self.A.batch_logits_to_list_font(tgt[:, 1:,1].cpu(),stopping_logits=end_token)
        self.log("val_pert/font_acc", font_acc(font_gt=gt_ft, font_pred=pred_ft), on_epoch=True, on_step=True,batch_size=tgt.shape[0],sync_dist=True )

        pred_char, pred_font, attention = self.forward(x=x, tgt=tgt[:,:-1], tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask[:,:-1])
        loss_char = self.criterion_char(pred_char, tgt[:,1:,0])
        self.log("val/loss_char",loss_char,batch_size=tgt.shape[0],sync_dist=True)
        loss_font = self.criterion_font(pred_font, tgt[:,1:,1])
        self.log("val/loss_font", loss_font,batch_size=tgt.shape[0],sync_dist=True)
        weight_char = 1.0
        weight_font = 1.0
        total_loss = (weight_char * loss_char) + (weight_font * loss_font)
        self.log("val/total_loss", total_loss, batch_size=tgt.shape[0],sync_dist=True)
        end_tokens = [self.A.toPosition['<eos>']]
        pred_str = self.A.batch_logits_to_string_list_font(torch.argmax(pred_char, dim=2).long().cpu(),
                                                                      stopping_logits=end_tokens)
        gt_str = self.A.batch_logits_to_string_list_font(tgt[:, 1:,0].cpu(),stopping_logits=end_tokens)
        self.log("val/cer", cer(str_gt=gt_str, str_pred=pred_str), on_epoch=True, on_step=True,batch_size=tgt.shape[0],sync_dist=True)
        end_token = [13]
        pred_ft = self.A.batch_logits_to_list_font(torch.argmax(pred_font, dim=2).long().cpu(),
                                                                stopping_logits=end_token)
        gt_ft = self.A.batch_logits_to_list_font(tgt[:, 1:,1].cpu(),stopping_logits=end_token)
        self.log("val/font_acc", font_acc(font_gt=gt_ft, font_pred=pred_ft), on_epoch=True, on_step=True,batch_size=tgt.shape[0],sync_dist=True )
        if batch_idx == 0:
            for idx in range(len(batch['LINE_IMAGE'])):
                if idx > 2:
                    break
                image = wandb.Image(batch['LINE_IMAGE'][idx],
                                    caption="GT:{} ||| PRED:{} ||| FONT GT:{} PRED:{}".format(
                                        gt_str[idx],
                                        pred_str[idx],
                                        tgt[:, 1:,1].cpu()[idx],
                                        torch.argmax(pred_font, dim=2).long().cpu()[idx],
                                        # torch.argmax(pred, dim=2).long().cpu()[idx],
                                    ))
                self.logger.experiment.log({"validation example {}".format(idx): image})
                att_img = attention[idx]
                att_img = att_img.reshape(att_img.shape[0], 2, -1)  # TODO: check 4 dynamically
                att_img = att_img.sum(dim=1)
                att_img -= att_img.min(1, keepdim=True)[0]
                att_img /= att_img.max(1, keepdim=True)[0]
                att_img_wandb = wandb.Image(att_img[:(~tgt_key_padding_mask[idx]).sum(), :].unsqueeze(0))
                self.logger.experiment.log({"validation attention example {}".format(idx): att_img_wandb})
        # torch.cuda.empty_cache()
        return total_loss
    
    
    def test_step(self,batch,batch_idx):
        x, tgt, tgt_key_padding_mask, tgt_mask = batch['LINE_IMAGE'], batch['S2S'], batch['TGT_KEY_PADDING_MASK'], batch['TGT_KEY_MASK']

        #Fair
        pred_char = torch.full((tgt.shape[0], tgt.shape[1]), self.A.toPosition['<pad>'], dtype=torch.long)
        # pred_char = (torch.ones(size=(tgt.shape[0], tgt.shape[1]))*self.A.toPosition['<pad>']).long()
        pred_font = (torch.zeros(size=(tgt.shape[0], tgt.shape[1]))).long()
        num_batches, num_timesteps, _ = tgt.shape 
        pred_c = torch.ones((num_batches, num_timesteps, len(self.A.toPosition)))
        pred_f = torch.ones((num_batches, num_timesteps, self.font_classes))
        # pred_char[:, 0] = self.A.toPosition['<sos>']
        pred_char[:,0] = (torch.ones(size=pred_char[:,0].shape)*self.A.toPosition['<sos>']).long()
        pred_font[:, 0] = (torch.ones(size=pred_font[:,0].shape)*0).long()
        pred_combined = torch.cat((pred_char.unsqueeze(-1), pred_font.unsqueeze(-1)), dim=-1)
        if self.on_gpu:
            pred_char = pred_char.cuda()
            pred_font = pred_font.cuda()
            pred_combined =  pred_combined.cuda()
            pred_c = pred_c.cuda()
            pred_f = pred_f.cuda()
        for i in range(1, tgt.shape[1]):
            out_char,out_font, attention = self.forward(x, tgt=pred_combined[:, :i])
            # pred_char[:, i] = torch.argmax(out_char, dim=2).long()[:, -1]
            # pred_font[:, i] = torch.argmax(out_font, dim=2).long()[:, -1]
            # pred_combined[:, i] = torch.cat((pred_char[:, i].unsqueeze(-1), pred_font[:, i].unsqueeze(-1)), dim=-1)
            pred_combined[:, i] = torch.cat((torch.argmax(out_char, dim=2).long()[:, -1].unsqueeze(-1), torch.argmax(out_font, dim=2).long()[:, -1].unsqueeze(-1)), dim=-1)
            pred_c[:,i-1] = out_char[:,-1,:]
            pred_f[:,i-1] = out_font[:,-1,:]
        loss_char = self.criterion_char(pred_c[:,:-1], tgt[:, 1:, 0])
        loss_font = self.criterion_font(pred_f[:,:-1], tgt[:, 1:, 1])
        self.log("test_fair/loss_char", loss_char,batch_size=tgt.shape[0],sync_dist=True)
        self.log("test_fair/loss_font", loss_font,batch_size=tgt.shape[0],sync_dist=True)
        weight_char = 1.0
        weight_font = 1.0
        total_loss = (weight_char * loss_char) + (weight_font * loss_font)
        self.log("test_fair/total_loss", total_loss,batch_size=tgt.shape[0],sync_dist=True)
        end_tokens = [self.A.toPosition['<eos>']]
        pred_str = self.A.batch_logits_to_string_list_font(torch.argmax(pred_c, dim=2).long().cpu(),
                                                                      stopping_logits=end_tokens)
        gt_str = self.A.batch_logits_to_string_list_font(tgt[:, 1:,0].cpu(),stopping_logits=end_tokens)
        self.log("test_fair/cer", cer(str_gt=gt_str, str_pred=pred_str), on_epoch=True, on_step=True,batch_size=tgt.shape[0])
        self.log("test_fair/wer", wer(str_gt=gt_str, str_pred=pred_str, seperators=self.A.seperators),batch_size=tgt.shape[0],on_epoch=True, on_step=True)
        end_token = [13]

        pred_ft = self.A.batch_logits_to_list_font(torch.argmax(pred_f, dim=2).long().cpu(),
                                                                stopping_logits=end_token)
        gt_ft = self.A.batch_logits_to_list_font(tgt[:, 1:,1].cpu(),stopping_logits=end_token)
        self.log("test_fair/font_acc", font_acc(font_gt=gt_ft, font_pred=pred_ft), on_epoch=True, on_step=True,batch_size=tgt.shape[0],sync_dist=True )
        self.log("test_fair/font_error_rate", int_cer(arr_gt=gt_ft, arr_pred=pred_ft), on_epoch=True, on_step=True,batch_size=tgt.shape[0],sync_dist=True )
        if batch_idx == 0:
            for idx in range(len(batch['LINE_IMAGE'])):
                if idx > 6:
                    break
                image = wandb.Image(batch['LINE_IMAGE'][idx],
                                    caption="GT:{} ||| PRED:{} ||| FONT GT:{} PRED:{}".format(
                                        gt_str[idx],
                                        pred_str[idx],
                                        gt_ft[idx],
                                        pred_ft[idx],
                                        # torch.argmax(pred, dim=2).long().cpu()[idx],
                                    ))
                self.logger.experiment.log({"test fair example {}".format(idx): image})
        # #Unfair
        pred_char, pred_font, attention = self.forward(x=x, tgt=tgt[:,:-1], tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask[:,:-1])
        loss_char = self.criterion_char(pred_char, tgt[:,1:,0])
        self.log("test_unfair/loss_char",loss_char,batch_size=tgt.shape[0])
        loss_font = self.criterion_font(pred_font, tgt[:,1:,1])
        self.log("test_unfair/loss_font", loss_font,batch_size=tgt.shape[0])
        weight_char = 1.0
        weight_font = 1.0
        total_loss = (weight_char * loss_char) + (weight_font * loss_font)
        self.log("test_unfair/total_loss", total_loss,batch_size=tgt.shape[0])
        end_tokens = [self.A.toPosition['<eos>']]
        pred_str = self.A.batch_logits_to_string_list_font(torch.argmax(pred_char, dim=2).long().cpu(),
                                                                      stopping_logits=end_tokens)
        gt_str = self.A.batch_logits_to_string_list_font(tgt[:, 1:,0].cpu(),stopping_logits=end_tokens)
        self.log("test_unfair/cer", cer(str_gt=gt_str, str_pred=pred_str), on_epoch=True, on_step=True,batch_size=tgt.shape[0])
        self.log("test_unfair/wer", wer(str_gt=gt_str, str_pred=pred_str, seperators=self.A.seperators),batch_size=tgt.shape[0],on_epoch=True, on_step=True)
        end_token = [13]
        pred_ft = self.A.batch_logits_to_list_font(torch.argmax(pred_font, dim=2).long().cpu(),
                                                                stopping_logits=end_token)
        gt_ft = self.A.batch_logits_to_list_font(tgt[:, 1:,1].cpu(),stopping_logits=end_token)
        self.log("test_unfair/font_acc", font_acc(font_gt=gt_ft, font_pred=pred_ft), on_epoch=True, on_step=True,batch_size=tgt.shape[0],sync_dist=True )
        self.log("test_unfair/font_error_rate", int_cer(arr_gt=gt_ft, arr_pred=pred_ft), on_epoch=True, on_step=True,batch_size=tgt.shape[0],sync_dist=True )
        if batch_idx == 0:
            for idx in range(len(batch['LINE_IMAGE'])):
                if idx > 6:
                    break
                image = wandb.Image(batch['LINE_IMAGE'][idx],
                                    caption="GT:{} ||| PRED:{} ||| FONT GT:{} PRED:{}".format(
                                        gt_str[idx],
                                        pred_str[idx],
                                        gt_ft[idx],
                                        pred_ft[idx],
                                        # torch.argmax(pred, dim=2).long().cpu()[idx],
                                    ))
                self.logger.experiment.log({"test unfair example {}".format(idx): image})
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
    


### Font Detection ###