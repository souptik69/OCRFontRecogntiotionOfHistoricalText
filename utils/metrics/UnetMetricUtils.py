import torch


def compute_pixel_accuracy(prediction, target):
    correct = (prediction == target).type(torch.uint8)
    return torch.sum(correct).item() / correct.nelement()


def compute_dice_coefficient(prediction, target):
    iflat = prediction.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    a_sum = torch.sum(iflat * iflat)
    b_sum = torch.sum(tflat * tflat)

    return (2. * intersection + 1) / (a_sum + b_sum + 1)
