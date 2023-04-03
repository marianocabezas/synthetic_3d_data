import time
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from torch.utils.data.dataset import Dataset
from utils import time_to_string


''' Datasets '''


class GeometricDataset(Dataset):
    def __init__(
            self, im_size, n_samples=1000, scale_ratio=3,
            min_back=100, max_back=200, fore_ratio=3, seed=None
    ):
        self.len = n_samples * 2
        self.max_x, self.max_y, self.max_z = im_size
        self.max_side = np.min(im_size)
        self.min_side = self.max_side / scale_ratio
        self.min_back = min_back
        self.max_back = max_back
        self.fore_ratio = fore_ratio
        self.seed = seed
        self.x, self.y, self.z = np.meshgrid(
            np.arange(0, self.max_x),
            np.arange(0, self.max_y),
            np.arange(0, self.max_z),
            indexing='ij'
        )

    def _coordinates(self):
        side_range = self.max_side - self.min_side
        x_offset = (2 * np.random.rand(1) - 1) * self.max_x * 0.25
        y_offset = (2 * np.random.rand(1) - 1) * self.max_y * 0.25
        z_offset = (2 * np.random.rand(1) - 1) * self.max_z * 0.25
        cx = 0.5 * self.max_x + x_offset
        cy = 0.5 * self.max_y + y_offset
        cz = 0.5 * self.max_z + z_offset
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

    def _cube_mask(self, x0, y0, z0, side, x=None, y=None, z=None):
        def side_mask(grid, coord):
            mask = np.logical_and(
                grid > (coord - side / 2), grid < (coord + side / 2)
            )
            return mask
        if x is None:
            x_mask = side_mask(self.x, x0)
        else:
            x_mask = side_mask(x, x0)
        if y is None:
            y_mask = side_mask(self.y, y0)
        else:
            y_mask = side_mask(y, y0)
        if z is None:
            z_mask = side_mask(self.z, z0)
        else:
            z_mask = side_mask(z, z0)
        mask = np.logical_and(
            np.logical_and(x_mask, y_mask), z_mask
        )
        return mask

    def _sphere_mask(self, a, b, c, radius, x=None, y=None, z=None):
        # (x - a)² + (y - b)² + (z - c)² = r²
        if x is None:
            x_dist = (self.x - a) ** 2
        else:
            x_dist = (x - a) ** 2
        if y is None:
            y_dist = (self.y - b) ** 2
        else:
            y_dist = (y - b) ** 2
        if z is None:
            z_dist = (self.z - c) ** 2
        else:
            z_dist = (z - c) ** 2
        mask = x_dist + y_dist + z_dist - radius ** 2 <= 0
        return mask

    def __len__(self):
        return self.len


class ShapesDataset(GeometricDataset):
    def __init__(
            self, im_size, n_samples=1000, scale_ratio=3,
            min_back=100, max_back=200, fore_ratio=3, seed=None
    ):
        # Init
        super().__init__(
            im_size, n_samples, scale_ratio, min_back, max_back, fore_ratio,
            seed
        )
        self.shapes = []
        self.masks = []
        self.labels = []
        init_start = time.time()
        if self.seed is not None:
            np.random.seed(self.seed)
            for i in range(self.len):
                time_elapsed = time.time() - init_start
                eta = self.len * time_elapsed / (i + 1)
                print(
                    '\033[KGenerating sample ({:d}/{:d}) {:} ETA {:}'.format(
                        i + 1, self.len,
                        time_to_string(time_elapsed),
                        time_to_string(eta),
                    ), end='\r'
                )
                cx, cy, cz, s = self._coordinates()
                background, foreground = self._gaussian_images()

                if i < n_samples:
                    # Cube
                    self.labels.append(0)
                    mask = self._cube_mask(cx, cy, cz, s)
                else:
                    # Sphere
                    self.labels.append(1)
                    mask = self._sphere_mask(cx, cy, cz, s / 2)

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
            if index < (self.len / 2):
                # Cube
                target_data = 0
                mask = self._cube_mask(cx, cy, cz, s)
            else:
                # Sphere
                target_data = 1
                mask = self._sphere_mask(cx, cy, cz, s / 2)

            data[mask] = foreground[mask]

        return np.expand_dims(data, axis=0), target_data


class LocationDataset(GeometricDataset):
    def __init__(
            self, im_size, n_samples=1000, scale_ratio=3,
            min_back=100, max_back=200, fore_ratio=3, seed=None
    ):
        # Init
        super().__init__(
            im_size, n_samples, scale_ratio, min_back, max_back, fore_ratio,
            seed
        )
        self.shapes = []
        self.masks = []
        self.labels = []
        init_start = time.time()
        if self.seed is not None:
            np.random.seed(self.seed)
            for i in range(self.len):
                time_elapsed = time.time() - init_start
                eta = self.len * time_elapsed / (i + 1)
                print(
                    '\033[KGenerating sample ({:d}/{:d}) {:} ETA {:}'.format(
                        i + 1, self.len,
                        time_to_string(time_elapsed),
                        time_to_string(eta),
                    ), end='\r'
                )
                cx, cy, cz, r = self._coordinates()
                background, foreground = self._gaussian_images()

                if i < n_samples:
                    # Sphere (top-left)
                    self.labels.append(0)
                    mask = self._sphere_mask(cx, cy, cz, r)
                else:
                    # Sphere (bottom-right)
                    self.labels.append(1)
                    mask = self._sphere_mask(
                        cx, cy, cz, r,
                        self.x + self.max_x / 2,
                        self.y + self.max_y / 2,
                        self.z + self.max_z / 2,
                    )

                background[mask] = foreground[mask]
                self.masks.append(mask)
                self.shapes.append(background)

    def _coordinates(self):
        side_range = self.max_side - self.min_side
        x_offset = (2 * np.random.rand(1) - 1) * self.max_x * 0.1
        y_offset = (2 * np.random.rand(1) - 1) * self.max_y * 0.1
        z_offset = (2 * np.random.rand(1) - 1) * self.max_z * 0.1
        cx = self.max_x * 0.25 + x_offset
        cy = self.max_y * 0.25 + y_offset
        cz = self.max_z * 0.25 + z_offset
        r = min(
            cx - 1, cy - 1, cz - 1,
            self.max_x * 0.5 - cx - 1,
            self.max_y * 0.5 - cy - 1,
            self.max_z * 0.5 - cz - 1,
            0.5 * (np.random.rand(1) * side_range + self.min_side)
        )

        return cx, cy, cz, r

    def __getitem__(self, index):
        if len(self.shapes) > 0:
            data = self.shapes[index]
            target_data = (self.labels[index], self.masks[index])
        else:
            cx, cy, cz, r = self._coordinates()
            data, foreground = self._gaussian_images()
            if index < (self.len / 2):
                # Sphere (top-left)
                target_data = 0
                mask = self._sphere_mask(cx, cy, cz, r)
            else:
                # Sphere (bottom-right)
                target_data = 1
                mask = self._sphere_mask(
                    cx, cy, cz, r,
                    self.x + self.max_x / 2,
                    self.y + self.max_y / 2,
                    self.z + self.max_z / 2,
                )

            data[mask] = foreground[mask]

        return np.expand_dims(data, axis=0), target_data


class ScaleDataset(GeometricDataset):
    def __init__(
            self, im_size, n_samples=1000, scale_ratio=3,
            min_back=100, max_back=200, fore_ratio=3, seed=None
    ):
        # Init
        super().__init__(
            im_size, n_samples, 1, min_back, max_back, fore_ratio,
            seed
        )
        self.scale = scale_ratio
        self.shapes = []
        self.masks = []
        self.labels = []
        init_start = time.time()
        if self.seed is not None:
            np.random.seed(self.seed)
            for i in range(self.len):
                time_elapsed = time.time() - init_start
                eta = self.len * time_elapsed / (i + 1)
                print(
                    '\033[KGenerating sample ({:d}/{:d}) {:} ETA {:}'.format(
                        i + 1, self.len,
                        time_to_string(time_elapsed),
                        time_to_string(eta),
                    ), end='\r'
                )
                cx, cy, cz, r = self._coordinates()
                background, foreground = self._gaussian_images()

                if i < n_samples:
                    # Big sphere
                    self.labels.append(0)
                    mask = self._sphere_mask(cx, cy, cz, r)
                else:
                    # Small sphere
                    self.labels.append(1)
                    mask = self._sphere_mask(cx, cy, cz, r / self.scale)

                background[mask] = foreground[mask]
                self.masks.append(mask)
                self.shapes.append(background)

    def _coordinates(self):
        x_offset = (2 * np.random.rand(1) - 1) * self.max_x * 0.25
        y_offset = (2 * np.random.rand(1) - 1) * self.max_y * 0.25
        z_offset = (2 * np.random.rand(1) - 1) * self.max_z * 0.25
        cx = self.max_x * 0.5 + x_offset
        cy = self.max_y * 0.5 + y_offset
        cz = self.max_z * 0.5 + z_offset
        r = np.random.normal(self.max_side * 0.25, self.max_side * 0.05)

        return cx, cy, cz, r

    def __getitem__(self, index):
        if len(self.shapes) > 0:
            data = self.shapes[index]
            target_data = (self.labels[index], self.masks[index])
        else:
            cx, cy, cz, r = self._coordinates()
            data, foreground = self._gaussian_images()
            if index < (self.len / 2):
                # Big sphere
                target_data = 0
                mask = self._sphere_mask(cx, cy, cz, r)
            else:
                # Small sphere
                target_data = 1
                mask = self._sphere_mask(cx, cy, cz, r / self.scale)

            data[mask] = foreground[mask]

        return np.expand_dims(data, axis=0), target_data


class RotationDataset(GeometricDataset):
    def __init__(
            self, im_size, n_samples=1000, scale_ratio=3,
            min_back=100, max_back=200, fore_ratio=3, seed=None
    ):
        # Init
        super().__init__(
            im_size, n_samples, scale_ratio, min_back, max_back, fore_ratio,
            seed
        )
        self.shapes = []
        self.masks = []
        self.labels = []
        init_start = time.time()
        if self.seed is not None:
            np.random.seed(self.seed)
            for i in range(self.len):
                time_elapsed = time.time() - init_start
                eta = self.len * time_elapsed / (i + 1)
                print(
                    '\033[KGenerating sample ({:d}/{:d}) {:} ETA {:}'.format(
                        i + 1, self.len,
                        time_to_string(time_elapsed),
                        time_to_string(eta),
                    ), end='\r'
                )
                cx, cy, cz, s = self._coordinates()
                background, foreground = self._gaussian_images()

                if i < n_samples:
                    # Normal cube
                    self.labels.append(0)
                    angle = np.random.normal(0, np.pi * 0.05)
                else:
                    # Rotated cube
                    self.labels.append(1)
                    angle = np.random.normal(np.pi / 4, np.pi * 0.05)

                x, y, z = self._rotate_grid(cx, cy, cz, angle)
                mask = self._cube_mask(x, y, z, cx, cy, cz, s)

                background[mask] = foreground[mask]
                self.masks.append(mask)
                self.shapes.append(background)

    def _rotate_grid(self, x0, y0, z0, angle):
        x_norm = self.x - x0
        y_norm = self.y - y0
        z_norm = self.z - z0
        cos = np.cos(angle)
        sin = np.sin(angle)
        rot_x = x_norm * cos + y_norm * sin * cos + z_norm * sin ** 2
        rot_y = - x_norm * sin + y_norm * cos ** 2 + z_norm * sin * cos
        rot_z = - y_norm * sin + z_norm * cos

        return rot_x + x0, rot_y + y0, rot_z + z0

    def __getitem__(self, index):
        if len(self.shapes) > 0:
            data = self.shapes[index]
            target_data = (self.labels[index], self.masks[index])
        else:
            cx, cy, cz, s = self._coordinates()
            data, foreground = self._gaussian_images()

            if index < (self.len / 2):
                # Normal cube
                target_data = 0
                angle = np.random.normal(0, np.pi * 0.01)
            else:
                # Rotated cube
                target_data = 1
                angle = np.random.normal(np.pi / 4, np.pi * 0.01)

            x, y, z = self._rotate_grid(cx, cy, cz, angle)
            mask = self._cube_mask(x, y, z, cx, cy, cz, s)

            data[mask] = foreground[mask]

        return np.expand_dims(data, axis=0), target_data


class GradientDataset(GeometricDataset):
    def __init__(
            self, im_size, n_samples=1000, scale_ratio=2,
            min_back=100, max_back=200, fore_ratio=3, seed=None
    ):
        # Init
        super().__init__(
            im_size, n_samples, scale_ratio, min_back, max_back, fore_ratio,
            seed
        )
        self.shapes = []
        self.masks = []
        self.labels = []

        init_start = time.time()
        if self.seed is not None:
            np.random.seed(self.seed)
            for i in range(self.len):
                time_elapsed = time.time() - init_start
                eta = self.len * time_elapsed / (i + 1)
                print(
                    '\033[KGenerating sample ({:d}/{:d}) {:} ETA {:}'.format(
                        i + 1, self.len,
                        time_to_string(time_elapsed),
                        time_to_string(eta),
                    ), end='\r'
                )
                cx, cy, cz, r = self._coordinates()
                background, foreground = self._gaussian_images()

                if i < n_samples:
                    # Normal sphere
                    self.labels.append(0)
                else:
                    # Concentric sphere
                    self.labels.append(1)
                    gradient = self._gradient(cx, cy, cz, r)
                    foreground = np.clip(
                        gradient * foreground,
                        0, np.max(foreground) - (self.max_back - self.min_back)
                    )

                mask = self._sphere_mask(cx, cy, cz, r)

                background[mask] = foreground[mask]
                self.masks.append(mask)
                self.shapes.append(background)

    def _gradient(self, cx, cy, cz, r):
        c_dist = (self.x - cx) ** 2 + (self.y - cy) ** 2 + (self.z - cz) ** 2
        gradient = np.clip(r ** 2 - c_dist, 0, r ** 2) / (r ** 2)

        return gradient

    def __getitem__(self, index):
        if len(self.shapes) > 0:
            data = self.shapes[index]
            target_data = (self.labels[index], self.masks[index])
        else:
            cx, cy, cz, r = self._coordinates()
            data, foreground = self._gaussian_images()

            if index < (self.len / 2):
                # Normal sphere
                target_data = 0
            else:
                # Concentric sphere
                target_data = 1
                gradient = self._gradient(cx, cy, cz, r)
                foreground = np.clip(
                    gradient * foreground,
                    0, np.max(foreground) - (self.max_back - self.min_back)
                )

            mask = self._sphere_mask(cx, cy, cz, r)

            data[mask] = foreground[mask] + data[mask]

        return np.expand_dims(data, axis=0), target_data


class ContrastDataset(GradientDataset):
    def __init__(
            self, im_size, n_samples=1000, scale_ratio=2,
            min_back=100, max_back=200, fore_ratio=3, seed=None
    ):
        # Init
        super().__init__(
            im_size, n_samples, scale_ratio, min_back, max_back, fore_ratio,
            seed
        )
        self.shapes = []
        self.masks = []
        self.labels = []

        init_start = time.time()
        if self.seed is not None:
            np.random.seed(self.seed)
            for i in range(self.len):
                time_elapsed = time.time() - init_start
                eta = self.len * time_elapsed / (i + 1)
                print(
                    '\033[KGenerating sample ({:d}/{:d}) {:} ETA {:}'.format(
                        i + 1, self.len,
                        time_to_string(time_elapsed),
                        time_to_string(eta),
                    ), end='\r'
                )
                cx, cy, cz, r = self._coordinates()
                background, foreground = self._gaussian_images()
                gradient = self._gradient(cx, cy, cz, r)

                if i < n_samples:
                    # Positive gradient
                    self.labels.append(0)
                    foreground = np.clip(
                        (1 - gradient) * foreground,
                        0, np.max(foreground) - (self.max_back - self.min_back)
                    )
                else:
                    # Negative gradient
                    self.labels.append(1)

                    foreground = np.clip(
                        gradient * foreground,
                        0, np.max(foreground) - (self.max_back - self.min_back)
                    )

                mask = self._sphere_mask(cx, cy, cz, r)

                background[mask] = foreground[mask]
                self.masks.append(mask)
                self.shapes.append(background)

    def __getitem__(self, index):
        if len(self.shapes) > 0:
            data = self.shapes[index]
            target_data = (self.labels[index], self.masks[index])
        else:
            cx, cy, cz, r = self._coordinates()
            data, foreground = self._gaussian_images()
            gradient = self._gradient(cx, cy, cz, r)

            if index < (self.len / 2):
                # Positive gradient
                target_data = 0
                foreground = np.clip(
                    (1 - gradient) * foreground,
                    0, np.max(foreground) - (self.max_back - self.min_back)
                )
            else:
                # Negative gradient
                target_data = 1
                foreground = np.clip(
                    gradient * foreground,
                    0, np.max(foreground) - (self.max_back - self.min_back)
                )

            mask = self._sphere_mask(cx, cy, cz, r)

            data[mask] = foreground[mask] + data[mask]

        return np.expand_dims(data, axis=0), target_data


class ParcellationDataset(ContrastDataset):
    def __init__(
            self, im_size, n_samples=1000, scale_ratio=2,
            min_back=100, max_back=200, fore_ratio=4,
            sigma=1, alpha=10, seed=None
    ):
        # Init
        self.sigma = sigma
        self.alpha = alpha
        super().__init__(
            im_size, n_samples, scale_ratio, min_back, max_back, fore_ratio,
            seed
        )

    def _gradient(self, cx, cy, cz, r):
        core_x, core_y, core_z = self._elastic_shape()
        core_d = (core_x - cx) ** 2 + (core_y - cy) ** 2 + (core_z - cz) ** 2
        core = (core_d < (r ** 2) / 9).astype(np.float32)
        mid_x, mid_y, mid_z = self._elastic_shape()
        mid_d = (mid_x - cx) ** 2 + (mid_y - cy) ** 2 + (mid_z - cz) ** 2
        mid = (mid_d < (r ** 2) * 4 / 9).astype(np.float32)
        bound_d = (self.x - cx) ** 2 + (self.y - cy) ** 2 + (self.z - cz) ** 2
        bound = (bound_d < (r ** 2)).astype(np.float32)
        gradient = (4 * core + 3 * mid + 2 * bound) / 7

        return gradient

    def _elastic_shape(self):
        dx = gaussian_filter(
            2 * np.random.rand(*self.x.shape) - 1,
            self.sigma, mode='constant', cval=0
        ) * self.alpha
        dy = gaussian_filter(
            2 * np.random.rand(*self.y.shape) - 1,
            self.sigma, mode='constant', cval=0
        ) * self.alpha
        dz = gaussian_filter(
            2 * np.random.rand(*self.z.shape) - 1,
            self.sigma, mode='constant', cval=0
        ) * self.alpha
        return self.x + dx, self.y + dy, self.z + dz

