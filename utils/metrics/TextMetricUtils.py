from typing import List
import numpy as np

import textdistance
import torch

def cer(str_gt: list, str_pred: list) -> torch.Tensor:
    len_, edit = 0, 0
    for pred, gt in zip(str_pred, str_gt):
        edit += textdistance.levenshtein(gt, pred)
        len_ += len(gt)
    cer = edit / len_
    return torch.Tensor([cer])


### Font Detection ###
# def font_acc(font_gt: list, font_pred: list) -> torch.Tensor:
#     len_,correct_count = 0,0
#     for gt,pred in zip(font_gt, font_pred):
#         correct_count += np.sum(np.isin(gt, pred))
#         len_ +=  len(gt)
#     accuracy = correct_count/len_
#     return torch.Tensor([accuracy]) 

def font_acc(font_gt, font_pred):
    total_correct = 0
    total_elements = 0

    for gt, pred in zip(font_gt, font_pred):
        min_length = min(len(gt), len(pred))
        correct = np.sum(gt[:min_length] == pred[:min_length])
        total_correct += correct
        total_elements += min_length

    accuracy = total_correct / total_elements
    return torch.tensor(accuracy)

def int_cer(arr_gt: np.ndarray, arr_pred: np.ndarray) -> torch.Tensor:
    len_, edit = 0, 0
    for pred, gt in zip(arr_pred, arr_gt):
        edit += textdistance.levenshtein(str(gt), str(pred))  # Convert integers to strings
        len_ += len(str(gt))
    cer = edit / len_
    return torch.Tensor([cer])
### Font detection ###


def wer(str_gt, str_pred, seperators = None):
    # TODO: check whether this list is complete
    if seperators is None:
        separation_marks = [' ', '.', ',', "'", '-', '"', '#', '(', ')', ':', ';', '?', '*', '!', '/', '&', '+']
    else:
        separation_marks = seperators
    # separation_marks = ["?", ".", ";", ",", "!", "\n"]
    len_ = 0
    edit = 0
    for pred, gt in zip(str_pred, str_gt):
        for mark in separation_marks:
            gt = gt.replace(mark, " ")
            pred = pred.replace(mark, " ")
        gt = gt.split(" ")
        pred = pred.split(" ")
        while '' in gt:
            gt.remove('')
        while '' in pred:
            pred.remove('')
        edit += textdistance.levenshtein(gt, pred)
        len_ += len(gt)
    cer = edit / len_
    return torch.min(torch.Tensor([cer, 1.]))

def compute_character_error_rate(preds: List[str], targets: List[str]):
    len_ = 0
    edit = 0
    for pred, target in zip(preds, targets):
        _, edit, len_ = compute_character_error_rate_single(pred, target)
    cer = edit / len_
    return torch.Tensor([cer])


def compute_character_error_rate_single(pred, target):
    edit, len_ = textdistance.levenshtein(target, pred), len(target)
    cer = edit / len_
    return torch.Tensor([cer]), edit, len_


def compute_word_error_rate(preds: List[str], targets: List[str]):
    len_ = 0
    edit = 0
    for pred, target in zip(preds, targets):
        _, edit_new, len_new = compute_word_error_rate_single(pred, target)
        edit += edit_new
        len_ += len_new
    cer = edit / len_
    return torch.Tensor([cer])


def compute_word_error_rate_single(pred: str, target: str):
    separation_marks = ["?", ".", ";", ",", "!", "\n"]
    for mark in separation_marks:
        target.replace(mark, " {} ".format(mark))
        pred.replace(mark, " {} ".format(mark))
    target = target.split(" ")
    pred = pred.split(" ")
    while '' in target:
        target.remove('')
    while '' in pred:
        pred.remove('')
    edit, len_ = textdistance.levenshtein(target, pred), len(target)
    cer = edit / len_
    return torch.Tensor([cer]), edit, len_
