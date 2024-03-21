from typing import List

import torch

from dataset.combined.CombinedLineInfo import CombinedLineInfo
from utils.alphabet import Alphabet
from utils.constants import *
from utils.SubSequentMask import subsequent_mask
import gc

####  Font Detection  ####


def collate_s2s_font(batch):
    keys = batch[0].keys()
    output = dict()
    A = Alphabet(dataset="NBB", mode="attention")
    ys =list()
    cf =list()
    for item in batch:
        logits= A.string_to_logits(item['TEXT'])
        cf_tensor = torch.LongTensor(item['CF'])
        max_length = max(logits.size(0), cf_tensor.size(0))
        logits = torch.cat([logits, torch.LongTensor([A.toPosition['<eos>']] * (max_length - logits.size(0)))])
        cf_tensor = torch.cat([cf_tensor, torch.LongTensor([13] * (max_length - cf_tensor.size(0)))])
        logits = torch.cat([torch.LongTensor([A.toPosition['<sos>']]), logits, torch.LongTensor([A.toPosition['<eos>']])])
        cf_tensor = torch.cat([torch.LongTensor([A.toPosition['<sos>']]), cf_tensor, torch.LongTensor([A.toPosition['<eos>']])])
        cf_tensor[0] = 0  
        cf_tensor[-1] = 13
        cf.append(cf_tensor)
        ys.append(logits)
    ys_padded = torch.nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=A.toPosition['<pad>'])
    ys_masking = torch.eq(ys_padded,torch.ones(ys_padded.shape,dtype=torch.long)*torch.LongTensor([A.toPosition['<pad>']]))
    # cf_padded = torch.nn.utils.rnn.pad_sequence(cf, batch_first=True, padding_value=A.toPosition['<pad>'])
    # cf_masking = torch.eq(cf_padded,torch.ones(cf_padded.shape,dtype=torch.long)*torch.LongTensor([A.toPosition['<pad>']]))
    cf_padded = torch.nn.utils.rnn.pad_sequence(cf, batch_first=True, padding_value=0)
    cf_masking = torch.eq(cf_padded, torch.zeros(cf_padded.shape, dtype=torch.long))
    ys_mask= subsequent_mask(ys_masking.shape[-1]-1)
    cf_mask = subsequent_mask(cf_masking.shape[-1]-1)
    max_seq_len = max(ys_padded.size(1), cf_padded.size(1))
    if ys_padded.size(1) < max_seq_len:
        ys_padded = torch.cat((ys_padded, torch.zeros(ys_padded.size(0), max_seq_len - ys_padded.size(1))), dim=1)
    if cf_padded.size(1) < max_seq_len:
        cf_padded = torch.cat((cf_padded, torch.zeros(cf_padded.size(0), max_seq_len - cf_padded.size(1))), dim=1)
    # joined_tensor = torch.cat((ys_padded.unsqueeze(2), cf_padded.unsqueeze(2)), dim=2)
    output['S2S'] = torch.cat((ys_padded.unsqueeze(2), cf_padded.unsqueeze(2)), dim=2)
    max_seq_len1 = max(ys_masking.size(1), cf_masking.size(1))
    if ys_masking.size(1) < max_seq_len1:
        ys_masking = torch.cat((ys_masking, torch.zeros(ys_masking.size(0), max_seq_len1 - ys_masking.size(1))), dim=1)
    if cf_masking.size(1) < max_seq_len1:
        cf_masking = torch.cat((cf_masking, torch.zeros(cf_masking.size(0), max_seq_len1 - cf_masking.size(1))), dim=1)
    # joined_tensor_1 = torch.cat((ys_masking.unsqueeze(2), cf_masking.unsqueeze(2)), dim=2)
    output['TGT_KEY_PADDING_MASK'] = ys_masking
    max_seq_len2 = max(ys_mask.size(1), cf_mask.size(1))
    if ys_mask.size(1) < max_seq_len2:
        padding_size = max_seq_len2 - ys_mask.size(1)
        pad_values = float('-inf')
        padding = torch.full((ys_mask.size(0), padding_size), pad_values, dtype=torch.float32)
        ys_mask = torch.cat((ys_mask, padding), dim=1)
    
    if cf_mask.size(1) < max_seq_len2:
        padding_size = max_seq_len2 - cf_mask.size(1)
        pad_values = float('-inf')
        padding = torch.full((cf_mask.size(0), padding_size), pad_values, dtype=torch.float32)
        cf_mask = torch.cat((cf_mask, padding), dim=1)
    max_batch_size = max(ys_mask.size(0), cf_mask.size(0))
    if ys_mask.size(0) < max_batch_size:
        ys_mask = torch.cat((ys_mask, torch.zeros(max_batch_size - ys_mask.size(0), ys_mask.size(1))), dim=0)
    if cf_mask.size(0) < max_batch_size:
        cf_mask = torch.cat((cf_mask, torch.zeros(max_batch_size - cf_mask.size(0), cf_mask.size(1))), dim=0)
    assert ys_mask.size(0) == cf_mask.size(0)
    # joined_tensor_2 = torch.cat((ys_mask.unsqueeze(2), cf_mask.unsqueeze(2)), dim=2)
    output['TGT_KEY_MASK'] = ys_mask
    for key in keys:
        if key == 'LINE_IMAGE':
            images = [item[key].permute(2, 0, 1) for item in batch]
            images_padded = torch.nn.utils.rnn.pad_sequence(images)
            output[key] = images_padded.permute(1, 2, 3, 0)
    return output


def collate_s2s_singlefont(batch):
    keys = batch[0].keys()
    output = dict()

    pf = list()
    for item in batch:
        pf_tensor = torch.LongTensor(item['PF'])
        pf_tensor = torch.cat([pf_tensor, torch.LongTensor([13])])
        pf.append(pf_tensor)
    pf_padded = torch.nn.utils.rnn.pad_sequence(pf, batch_first=True, padding_value=0)
    output['S2S'] = pf_padded
    pf_masking = torch.eq(pf_padded, torch.zeros(pf_padded.shape, dtype=torch.long))
    output['TGT_KEY_PADDING_MASK'] = pf_masking
    output['TGT_KEY_MASK'] = subsequent_mask(pf_masking.shape[-1]-1)

    for key in keys:
        if key == 'LINE_IMAGE':
            images = [item[key].permute(2, 0, 1) for item in batch]
            images_padded = torch.nn.utils.rnn.pad_sequence(images)
            output[key] = images_padded.permute(1, 2, 3, 0)

    return output

####  Font Detection ####


def collate_s2s(batch):
    keys = batch[0].keys()
    output = dict()
    A = Alphabet(dataset="NBB", mode="s2s_recipient")

    ys = list()
    for item in batch:
        logits = A.string_to_logits(item[TEXT])
        if item[RECIPIENT]:
            logits = torch.cat([torch.LongTensor([A.toPosition[START_OF_SEQUENCE]]), logits, torch.LongTensor([A.toPosition[END_OF_SEQUENCE_RECIPIENT]])])
        else:
            logits = torch.cat([torch.LongTensor([A.toPosition[START_OF_SEQUENCE]]), logits, torch.LongTensor([A.toPosition[END_OF_SEQUENCE_BODY]])])
        ys.append(logits)
    ys_padded = torch.nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=A.toPosition[PAD])
    output[S2S_TEXT] = ys_padded
    ys_masking = torch.eq(ys_padded,torch.ones(ys_padded.shape,dtype=torch.long)*torch.LongTensor([A.toPosition[PAD]]))
    output[TGT_KEY_PADDING_MASK] = ys_masking
    output[TGT_MASK] = subsequent_mask(ys_masking.shape[-1]-1)

    for key in keys:
        if key == LINE_IMAGE:
            images = [item[key].permute(2, 0, 1) for item in batch]
            images_padded = torch.nn.utils.rnn.pad_sequence(images)
            output[key] = images_padded.permute(1, 2, 3, 0)

    return output

def collate_s2s_both(batch):
    keys = batch[0].keys()
    output = {}

    for idx, key in enumerate(keys):
        if key == IMAGE:
            val = [item[key] for item in batch]
            output[key] = torch.stack(val)
        elif key == MASK:
            val = [item[key] for item in batch]
            output[key] = torch.stack(val)
        elif key == "original_height":
            val = [item[key] for item in batch]
            output[key] = torch.Tensor(val)
        elif key == "original_width":
            val = [item[key] for item in batch]
            output[key] = torch.Tensor(val)
        elif key == "content_infos":
            val = [item[key] for item in batch]
            output[key] = val
        elif key == "line_batch":
            val = [item[key] for item in batch]
            output[key] = val
        elif key == "original_image":
            val = [item[key] for item in batch]
            output[key] = val
        else:
            raise NotImplementedError("forgot key: ",key)

    return output


def collate_unet(batch):
    keys = batch[0].keys()
    output = {}

    for idx, key in enumerate(keys):
        if key == IMAGE:
            val = [item[key] for item in batch]
            output[key] = torch.stack(val)
        elif key == MASK:
            val = [item[key] for item in batch]
            output[key] = torch.stack(val)
        elif key == "original_height":
            val = [item[key] for item in batch]
            output[key] = torch.Tensor(val)
        elif key == "original_width":
            val = [item[key] for item in batch]
            output[key] = torch.Tensor(val)
        elif key == "content_infos":
            val = [item[key] for item in batch]
            output[key] = val

    return output


def collate_with_padded_width(batch):
    keys = batch[0].keys()
    output = {
        UNPADDED_IMAGE_WIDTH: [item[IMAGE].shape[2] for item in batch],
        TEXT: [item[TEXT] for item in batch]
    }

    for idx, key in enumerate(keys):
        if key == IMAGE:
            images = [item[key].permute(2, 0, 1) for item in batch]
            images_padded = torch.nn.utils.rnn.pad_sequence(images)
            output[key] = images_padded.permute(1, 2, 3, 0)
        elif key in [UNPADDED_TEXT_LEN]:
            val = [item[key] for item in batch]
            output[key] = torch.tensor(val)
       # elif key == TEXT:
       #     output[key] = torch.stack([item[key] for item in batch])
        elif key == RECIPIENT:
            output[RECIPIENT] = torch.Tensor([int(item[RECIPIENT]) for item in batch])
    return output


def collate_combined(batch):
    # we can be sure that the batch size is 1 in this case
    item = batch[0]

    output = {
        IMAGE: item[IMAGE],
        MASK: item[MASK],
        ORIGINAL_IMAGE_SIZE: item[ORIGINAL_IMAGE_SIZE],
        ORIGINAL_IMAGE: item[ORIGINAL_IMAGE]
    }

    line_infos: List[CombinedLineInfo] = item[CONTENT_INFOS]

    if len(line_infos) > 0:

        # pad images
        line_images = [line_info.line_image.permute(2, 0, 1) for line_info in line_infos]
        line_images_padded = torch.nn.utils.rnn.pad_sequence(line_images).permute(1, 2, 3, 0)

        for idx, line_info in enumerate(line_infos):
            line_info.line_image = line_images_padded[idx]

    output[CONTENT_INFOS] = line_infos
    return output
