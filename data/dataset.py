import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from data.preprocessing import MedicalImagePreprocessor


def apply_elastic_transform(image, mask, alpha=30, sigma=5):
    shape = image.shape[:2]
    dx = (
        cv2.GaussianBlur(
            (np.random.rand(*shape) * 2 - 1).astype(np.float32), (0, 0), sigma
        )
        * alpha
    )
    dy = (
        cv2.GaussianBlur(
            (np.random.rand(*shape) * 2 - 1).astype(np.float32), (0, 0), sigma
        )
        * alpha
    )

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    map_x = np.float32(x + dx)
    map_y = np.float32(y + dy)

    def_img = cv2.remap(
        image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
    )
    def_mask = cv2.remap(
        mask, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT
    )
    return def_img, def_mask


def apply_glare_simulation(image):
    if random.random() > 0.5:
        h, w = image.shape[:2]
        center_x = random.randint(w // 4, 3 * w // 4)
        center_y = random.randint(h // 4, 3 * h // 4)
        radius = random.randint(min(h, w) // 8, min(h, w) // 3)

        y_grid, x_grid = np.ogrid[:h, :w]
        dist = np.sqrt((x_grid - center_x) ** 2 + (y_grid - center_y) ** 2)

        glare_mask = np.clip(1.0 - dist / radius, 0, 1) ** 2
        glare_intensity = random.randint(40, 100)

        img_float = image.astype(np.float32) + glare_mask * glare_intensity
        return np.clip(img_float, 0, 255).astype(np.uint8)
    return image


class SyntheticElongatedStructureGenerator:
    def __init__(self, image_size=(128, 128), num_structures=3):
        self.image_size = image_size
        self.num_structures = num_structures

    def generate(self):
        h, w = self.image_size
        image = np.zeros((h, w), dtype=np.uint8)
        mask = np.zeros((h, w), dtype=np.uint8)

        noise = np.random.normal(100, 25, (h, w)).clip(0, 255).astype(np.uint8)
        image = cv2.add(image, noise)

        for _ in range(self.num_structures):
            is_horizontal = random.choice([True, False])
            thickness = random.randint(1, 3)

            if is_horizontal:
                y = random.randint(10, h - 10)
                x_start = random.randint(5, w // 3)
                x_end = random.randint(2 * w // 3, w - 5)

                pts = []
                num_pts = random.randint(4, 7)
                xs = np.linspace(x_start, x_end, num_pts, dtype=np.int32)
                for x in xs:
                    offset = random.randint(-8, 8)
                    pts.append([x, max(0, min(h - 1, y + offset))])

                pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(
                    mask,
                    [pts],
                    isClosed=False,
                    color=255,
                    thickness=thickness,
                )
                cv2.polylines(
                    image,
                    [pts],
                    isClosed=False,
                    color=220,
                    thickness=thickness,
                )
            else:
                x = random.randint(10, w - 10)
                y_start = random.randint(5, h // 3)
                y_end = random.randint(2 * h // 3, h - 5)

                pts = []
                num_pts = random.randint(4, 7)
                ys = np.linspace(y_start, y_end, num_pts, dtype=np.int32)
                for y in ys:
                    offset = random.randint(-8, 8)
                    pts.append([max(0, min(w - 1, x + offset)), y])

                pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(
                    mask,
                    [pts],
                    isClosed=False,
                    color=255,
                    thickness=thickness,
                )
                cv2.polylines(
                    image,
                    [pts],
                    isClosed=False,
                    color=220,
                    thickness=thickness,
                )

        image = cv2.GaussianBlur(image, (3, 3), 0)
        mask = (mask > 127).astype(np.uint8)
        return image, mask


class ElongatedStructureDataset(Dataset):
    def __init__(
        self,
        image_paths=None,
        mask_paths=None,
        length=100,
        image_size=(128, 128),
        use_clahe=True,
        augment=False,
    ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.length = length if image_paths is None else len(image_paths)
        self.image_size = image_size
        self.augment = augment
        self.preprocessor = MedicalImagePreprocessor(
            use_clahe=use_clahe, norm_mode="minmax"
        )
        self.synthetic_generator = SyntheticElongatedStructureGenerator(
            image_size=image_size
        )

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.image_paths is not None and self.mask_paths is not None:
            image = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, self.image_size)
            mask = cv2.resize(
                mask, self.image_size, interpolation=cv2.INTER_NEAREST
            )
            mask = (mask > 127).astype(np.uint8)
        else:
            image, mask = self.synthetic_generator.generate()

        if self.augment:
            if random.random() > 0.5:
                image = cv2.flip(image, 1)
                mask = cv2.flip(mask, 1)
            if random.random() > 0.5:
                image = cv2.flip(image, 0)
                mask = cv2.flip(mask, 0)
            if random.random() > 0.5:
                image, mask = apply_elastic_transform(image, mask)
            image = apply_glare_simulation(image)

        img_tensor = self.preprocessor.process(image)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()

        return img_tensor, mask_tensor
