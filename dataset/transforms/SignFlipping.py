from PIL import Image, ImageOps


class SignFlipping:
    """
    Color inversion
    """

    def __init__(self):
        pass

    def __call__(self, x):
        return ImageOps.invert(x)
