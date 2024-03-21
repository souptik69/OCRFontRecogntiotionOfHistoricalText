import torchvision.transforms.functional as F


class FixedHeightResize:
    def __init__(self, height):
        self.height = height

    def __call__(self, img):
        old_width, old_height = img.size
        size = (self.height, self.calc_new_width(self.height, old_height, old_width))
        return F.resize(img, size)

    @staticmethod
    def calc_new_width(new_height, old_height, old_width) -> int:
        aspect_ratio = old_width / old_height
        return round(new_height * aspect_ratio)
