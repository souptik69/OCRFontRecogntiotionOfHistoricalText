import torch.nn as nn


class FixedWidthPadding:
    def __init__(self, width):
        self.width = width

    def __call__(self, img):
        width_diff = self.width - img.shape[2]
        m = nn.ConstantPad2d((0, width_diff, 0, 0), 0)
        img = m(img)
        return img

    @staticmethod
    def calc_new_width(new_height, old_height, old_width) -> int:
        aspect_ratio = old_width / old_height
        return round(new_height * aspect_ratio)
