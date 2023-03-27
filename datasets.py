import numpy as np
from torch.utils.data.dataset import Dataset
from utils import get_bb


''' Datasets '''


class ImageDataset(Dataset):
    def __init__(
        self, subjects, labels, rois
    ):
        # Init
        self.subjects = subjects
        self.rois = rois
        self.labels = labels

    def __getitem__(self, index):
        data = self.subjects[index]
        labels = self.labels[index]
        none_slice = (slice(None, None),)
        bb = get_bb(self.rois[index], 1)
        # Patch "extraction".
        if isinstance(data, tuple):
            data = tuple(
                data_i[none_slice + bb].astype(np.float32)
                for data_i in data
            )
        else:
            data = data[none_slice + bb].astype(np.float32)
        target_data = np.expand_dims(labels[bb].astype(np.uint8), axis=0)

        return data, target_data

    def __len__(self):
        return len(self.labels)


class MultiDataset(Dataset):
    """
    Dataset that combines multiple datasets into one.
    """

    def __init__(self, datasets):
        self.datasets = datasets
        self.lengths = np.cumsum([len(d) for d in self.datasets])

    def __getitem__(self, index):
        set_index = np.min(np.where(self.lengths > index))
        lengths = [0] + self.lengths.tolist()
        true_index = index - lengths[set_index]
        return self.datasets[set_index][true_index]

    def __len__(self):
        return self.lengths[-1]
