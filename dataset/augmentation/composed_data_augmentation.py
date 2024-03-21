import numpy as np
from PIL import Image
from torchvision.transforms import RandomPerspective
from torchvision.transforms.functional import adjust_contrast, adjust_brightness

from dataset.augmentation.augmentation_params import DA_CONFIG
from dataset.augmentation.Dilation import Dilation
from dataset.augmentation.DPIAdjusting import DPIAdjusting
from dataset.augmentation.Erosion import Erosion
from dataset.augmentation.ElasticDistortion import ElasticDistortion
from dataset.augmentation.RandomTransform import RandomTransform

class ComposedDataAugmentation:

    def __init__(self):
        self.da_config = DA_CONFIG

    def __call__(self, img):

        if "dpi" in self.da_config.keys() and np.random.rand() < self.da_config["dpi"]["proba"]:
            valid_factor = False
            while not valid_factor:
                factor = np.random.uniform(self.da_config["dpi"]["min_factor"], self.da_config["dpi"]["max_factor"])
                valid_factor = True
                if ("max_width" in self.da_config["dpi"].keys() and factor * img.size[0] > self.da_config["dpi"]["max_width"]) or \
                        ("max_height" in self.da_config["dpi"].keys() and factor * img.size[1] > self.da_config["dpi"][
                            "max_height"]):
                    valid_factor = False
                if ("min_width" in self.da_config["dpi"].keys() and factor * img.size[0] < self.da_config["dpi"]["min_width"]) or \
                        ("min_height" in self.da_config["dpi"].keys() and factor * img.size[1] < self.da_config["dpi"][
                            "min_height"]):
                    valid_factor = False
            img = DPIAdjusting(factor)(img)

        if "perspective" in self.da_config.keys() and np.random.rand() < self.da_config["perspective"]["proba"]:
            scale = np.random.uniform(self.da_config["perspective"]["min_factor"], self.da_config["perspective"]["max_factor"])
            img = RandomPerspective(distortion_scale=scale, p=1, interpolation=Image.BILINEAR, fill=0)(img)

        elif "elastic_distortion" in self.da_config.keys() and np.random.rand() < self.da_config["elastic_distortion"]["proba"]:
            magnitude = np.random.randint(1, self.da_config["elastic_distortion"]["max_magnitude"] + 1)
            kernel = np.random.randint(1, self.da_config["elastic_distortion"]["max_kernel"] + 1)
            magnitude_w, magnitude_h = (magnitude, 1) if np.random.randint(2) == 0 else (1, magnitude)
            img = ElasticDistortion(grid=(kernel, kernel), magnitude=(magnitude_w, magnitude_h), min_sep=(1, 1))(img)

        elif "random_transform" in self.da_config.keys() and np.random.rand() < self.da_config["random_transform"]["proba"]:
            img = RandomTransform(self.da_config["random_transform"]["max_val"])(img)

        if "dilation_erosion" in self.da_config.keys() and np.random.rand() < self.da_config["dilation_erosion"]["proba"]:
            kernel_h = np.random.randint(self.da_config["dilation_erosion"]["min_kernel"],
                                         self.da_config["dilation_erosion"]["max_kernel"] + 1)
            kernel_w = np.random.randint(self.da_config["dilation_erosion"]["min_kernel"],
                                         self.da_config["dilation_erosion"]["max_kernel"] + 1)
            if np.random.randint(2) == 0:
                img = Erosion((kernel_w, kernel_h), self.da_config["dilation_erosion"]["iterations"])(img)
            else:
                img = Dilation((kernel_w, kernel_h), self.da_config["dilation_erosion"]["iterations"])(img)

        if "contrast" in self.da_config.keys() and np.random.rand() < self.da_config["contrast"]["proba"]:
            factor = np.random.uniform(self.da_config["contrast"]["min_factor"], self.da_config["contrast"]["max_factor"])
            img = adjust_contrast(img, factor)

        if "brightness" in self.da_config.keys() and np.random.rand() < self.da_config["brightness"]["proba"]:
            factor = np.random.uniform(self.da_config["brightness"]["min_factor"], self.da_config["brightness"]["max_factor"])
            img = adjust_brightness(img, factor)

        return img

if __name__ == "__main__":
    cda = ComposedDataAugmentation()
