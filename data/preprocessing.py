import cv2
import numpy as np
import torch


class CLAHEPreprocessor:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size
        )

    def apply(self, image):
        if image.ndim == 3 and image.shape[2] == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            l_clahe = self.clahe.apply(l)
            lab_clahe = cv2.merge((l_clahe, a, b))
            return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
        elif image.ndim == 2:
            return self.clahe.apply(image)
        else:
            return image


class IntensityNormalizer:
    def __init__(self, mode="minmax"):
        self.mode = mode

    def normalize(self, image):
        image = image.astype(np.float32)
        if self.mode == "minmax":
            min_val = np.min(image)
            max_val = np.max(image)
            if max_val - min_val > 1e-8:
                return (image - min_val) / (max_val - min_val)
            return np.zeros_like(image)
        elif self.mode == "zscore":
            mean = np.mean(image)
            std = np.std(image)
            if std > 1e-8:
                return (image - mean) / std
            return image - mean
        return image


class MedicalImagePreprocessor:
    def __init__(self, use_clahe=True, norm_mode="minmax", clip_limit=2.0):
        self.use_clahe = use_clahe
        self.clahe = CLAHEPreprocessor(clip_limit=clip_limit)
        self.normalizer = IntensityNormalizer(mode=norm_mode)

    def process(self, image):
        if self.use_clahe:
            if image.dtype != np.uint8:
                img_uint8 = (
                    (image - image.min())
                    / (image.max() - image.min() + 1e-8)
                    * 255
                ).astype(np.uint8)
            else:
                img_uint8 = image
            image = self.clahe.apply(img_uint8)

        normalized = self.normalizer.normalize(image)

        if normalized.ndim == 2:
            tensor = torch.from_numpy(normalized).unsqueeze(0).float()
        elif normalized.ndim == 3:
            tensor = torch.from_numpy(normalized).permute(2, 0, 1).float()
        else:
            tensor = torch.from_numpy(normalized).float()

        return tensor
