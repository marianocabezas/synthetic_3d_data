import numpy as np
from torch.utils.data.dataset import Dataset
from utils import get_bb


''' Datasets '''


class GeometricDataset(Dataset):
    def __init__(
            self, im_size, n_samples=1000, scale_ratio=10,
            min_back=100, max_back=200, fore_ratio=1.5, seed=None
    ):
        self.len = n_samples * 2
        self.max_x, self.max_y, self.max_z = im_size
        self.max_side = np.min(im_size) / 2
        self.min_side = self.max_side / scale_ratio
        self.min_back = min_back
        self.max_back = max_back
        self.fore_ratio = fore_ratio
        self.seed = seed

    def _coordinates(self):
        side_range = self.max_side - self.min_side
        cx = np.random.rand(1) * 0.5 * self.max_x + (self.max_x * 0.25)
        cy = np.random.rand(1) * 0.5 * self.max_y + (self.max_y * 0.25)
        cz = np.random.rand(1) * 0.5 * self.max_z + (self.max_z * 0.25)
        s = np.random.rand(1) * side_range + self.min_side

        return cx, cy, cz, s

    def _gaussian_images(self):
        im_size = (self.max_x, self.max_y, self.max_z)
        bck_range = self.max_back - self.min_back
        mean_back = np.random.rand(1) * bck_range + self.min_back
        std_value = 0.5 * mean_back
        mean_foreground = mean_back * self.fore_ratio

        background = np.random.normal(mean_back, std_value, im_size)
        foreground = np.random.normal(mean_foreground, std_value, im_size)

        return background, foreground

    def _image_grid(self):
        return np.meshgrid(
            np.arange(0, self.max_x),
            np.arange(0, self.max_y),
            np.arange(0, self.max_z),
            indexing='ij'
        )

    def _cube_mask(self, x, y, z, x0, y0, z0, side):
        x_mask = np.logical_and(x > x0, x < (x0 + side))
        y_mask = np.logical_and(y > y0, y < (y0 + side))
        z_mask = np.logical_and(z > z0, z < (z0 + side))
        mask = np.logical_and(
            np.logical_and(x_mask, y_mask), z_mask
        )
        return mask

    def _sphere_mask(self, x, y, z, a, b, c, radius):
        # (x - a)² + (y - b)² + (z - c)² = r²
        mask = (x - a) ** 2 + (y - b) ** 2 + (z - c) ** 2 - radius ** 2 <= 0
        return mask

    def __len__(self):
        return self.len


class ShapesDataset(GeometricDataset):
    def __init__(
        self, im_size, n_samples=1000, scale_ratio=10,
        min_back=100, max_back=200, fore_ratio=1.5, seed=None
    ):
        # Init
        super().__init__(
            im_size, n_samples, scale_ratio, min_back, max_back, fore_ratio,
            seed
        )
        self.shapes = []
        self.masks = []
        self.labels = []
        if self.seed is not None:
            np.random.seed(self.seed)
            for i in range(self.len):
                cx, cy, cz, s = self._coordinates()
                background, foreground = self._gaussian_images()
                x, y, z = self._image_grid()

                if i < n_samples:
                    # Cubes
                    self.labels.append(0)
                    mask = self._cube_mask(x, y, z, cx, cy, cz, s)
                else:
                    # Spheres
                    self.labels.append(1)
                    mask = self._sphere_mask(x, y, z, cx, cy, cz, s / 2)

                background[mask] = foreground[mask]
                self.masks.append(mask)
                self.shapes.append(background)

    def __getitem__(self, index):
        if len(self.shapes) > 0:
            data = self.shapes[index]
            target_data = (self.labels[index], self.masks[index])
        else:
            cx, cy, cz, s = self._coordinates()
            data, foreground = self._gaussian_images()
            x, y, z = self._image_grid()
            if index < (self.len / 2):
                # Cubes
                target_data = 0
                mask = self._cube_mask(x, y, z, cx, cy, cz, s)
            else:
                # Spheres
                target_data = 1
                mask = self._sphere_mask(x, y, z, cx, cy, cz, s / 2)

            data[mask] = foreground[mask]

        return data, target_data
