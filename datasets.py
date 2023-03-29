import numpy as np
from torch.utils.data.dataset import Dataset


''' Datasets '''


class GeometricDataset(Dataset):
    def __init__(
            self, im_size, n_samples=1000, scale_ratio=2.5,
            min_back=100, max_back=200, fore_ratio=2, seed=None
    ):
        self.len = n_samples * 2
        self.max_x, self.max_y, self.max_z = im_size
        self.max_side = np.min(im_size)
        self.min_side = self.max_side / scale_ratio
        self.min_back = min_back
        self.max_back = max_back
        self.fore_ratio = fore_ratio
        self.seed = seed

    def _coordinates(self):
        side_range = self.max_side - self.min_side
        cx = 0.5 * self.max_x + (np.random.rand(1) * self.max_x * 0.25)
        cy = 0.5 * self.max_y + (np.random.rand(1) * self.max_y * 0.25)
        cz = 0.5 * self.max_z + (np.random.rand(1) * self.max_z * 0.25)
        s = min(
            cx - 1, cy - 1, cz - 1,
            self.max_x - cx - 1,
            self.max_y - cy - 1,
            self.max_z - cz - 1,
            np.random.rand(1) * side_range + self.min_side
        )

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
        x_mask = np.logical_and(
            x > (x0 - side / 2), x < (x0 + side / 2)
        )
        y_mask = np.logical_and(
            y > (y0 - side / 2), y < (y0 + side / 2)
        )
        z_mask = np.logical_and(
            z > (z0 - side / 2), z < (z0 + side / 2)
        )
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
            self, im_size, n_samples=1000, scale_ratio=2.5,
            min_back=100, max_back=200, fore_ratio=2, seed=None
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


class LocationDataset(GeometricDataset):
    def __init__(
            self, im_size, n_samples=1000, scale_ratio=2.5,
            min_back=100, max_back=200, fore_ratio=2, seed=None
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
                cx, cy, cz, r = self._coordinates()
                background, foreground = self._gaussian_images()
                x, y, z = self._image_grid()

                if i < n_samples:
                    # Spheres (top-left)
                    self.labels.append(0)
                    mask = self._sphere_mask(x, y, z, cx, cy, cz, r)
                else:
                    # Spheres (bottom-right)
                    self.labels.append(1)
                    mask = self._sphere_mask(
                        x, y, z,
                        cx + self.max_x / 2,
                        cy + self.max_y / 2,
                        cz + self.max_z / 2,
                        r
                    )

                background[mask] = foreground[mask]
                self.masks.append(mask)
                self.shapes.append(background)

    def _coordinates(self):
        side_range = self.max_side - self.min_side
        cx = self.max_x * 0.25 + np.random.rand(1) * self.max_x * 0.1
        cy = self.max_y * 0.25 + np.random.rand(1) * self.max_y * 0.1
        cz = self.max_z * 0.25 + np.random.rand(1) * self.max_z * 0.1
        r = min(
            cx - 1, cy - 1, cz - 1,
            self.max_x * 0.5 - cx - 1,
            self.max_y * 0.5 - cy - 1,
            self.max_z * 0.5 - cz - 1,
            np.random.rand(1) * side_range + self.min_side
        )

        return cx, cy, cz, r

    def __getitem__(self, index):
        if len(self.shapes) > 0:
            data = self.shapes[index]
            target_data = (self.labels[index], self.masks[index])
        else:
            cx, cy, cz, r = self._coordinates()
            data, foreground = self._gaussian_images()
            x, y, z = self._image_grid()
            if index < (self.len / 2):
                # Spheres (top-left)
                target_data = 0
                mask = self._sphere_mask(x, y, z, cx, cy, cz, r)
            else:
                # Spheres (bottom-right)
                target_data = 1
                mask = self._sphere_mask(
                    x, y, z,
                    cx + self.max_x / 2,
                    cy + self.max_y / 2,
                    cz + self.max_z / 2,
                    r
                )

            data[mask] = foreground[mask]

        return data, target_data
