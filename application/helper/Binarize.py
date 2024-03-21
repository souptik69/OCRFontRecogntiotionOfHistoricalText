import numpy as np
from PIL import Image
from skimage.filters import threshold_sauvola


class Binarize:

    def __call__(self, x):
        x_numpy = np.array(x)
        bin_img = (x_numpy < threshold_sauvola(x_numpy, window_size=55)).astype(np.uint8)
        img = Image.fromarray(bin_img * 255)
        return img
