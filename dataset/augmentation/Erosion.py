import numpy as np
from PIL import Image
from cv2 import erode


class Erosion:
    """
    OCR: stroke width decreasing
    input type: Pillow Image
    """

    def __init__(self, kernel, iterations):
        self.kernel = np.ones(kernel, np.uint8)
        self.iterations = iterations

    def __call__(self, x):
        return Image.fromarray(erode(np.array(x), self.kernel, iterations=self.iterations))
