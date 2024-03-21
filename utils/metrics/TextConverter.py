from utils.constants import PAD, END_OF_SEQUENCE
import torch


def convert_vector_to_text(encoded_text, lookup_alphabet):
    text = ""
    for sign_value in encoded_text:
        char = lookup_alphabet[sign_value.item()]

        if PAD == char or END_OF_SEQUENCE == char:
            break
        text = text + char
    return text


def ctc_remove_successives_identical_ind(ind, remove_ind=None, pad_ind=None):
    res = []
    for i in ind:
        if res and res[-1] == i:
            continue
        res.append(i)
    out = []
    for r in res:
        if remove_ind is not None and r in remove_ind:
            continue
        if pad_ind is not None and r == pad_ind:
            break
        out.append(r)
    return torch.tensor(out)

def ctc_remove_successives_identical_ind_batch(ind_batch, remove_ind=None, pad_ind=None):
    res = []
    for i_batch in ind_batch:
        res.append(ctc_remove_successives_identical_ind(i_batch, remove_ind, pad_ind))
    return res