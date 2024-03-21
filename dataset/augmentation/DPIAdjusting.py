import numpy as np
from PIL import Image


class DPIAdjusting:
    """
    Resolution modification
    """

    def __init__(self, factor):
        self.factor = factor

    def __call__(self, x):
        w, h = x.size
        return x.resize((int(np.ceil(w * self.factor)), int(np.ceil(h * self.factor))), Image.BILINEAR)
