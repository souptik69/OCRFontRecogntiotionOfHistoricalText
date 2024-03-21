import numpy as np
from PIL import Image
from cv2 import dilate

class Dilation:
    """
    OCR: stroke width increasing
    """

    def __init__(self, kernel, iterations):
        self.kernel = np.ones(kernel, np.uint8)
        self.iterations = iterations

    def __call__(self, x):
        return Image.fromarray(dilate(np.array(x), self.kernel, iterations=self.iterations))
