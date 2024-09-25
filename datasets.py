import numpy as np
import torch.utils.data as data


class NumpyDataset(data.Dataset):

    def __init__(self, X, y, transform=None, output_labels=True):
        self.X = X
        self.y = y
        self.transform = transform
        self.output_labels = output_labels

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        x = self.X[index]
        if self.transform:
            x = self.transform(x)
        x = x.astype(np.float32)
        if not self.output_labels:
            return x
        y = self.y[index]
        y = y.astype(np.int64)
        return x, y

    def weighted_sampler(self):
        labels, counts = np.unique(self.y, return_counts=True)
        class_weights = 1 / counts  # Inverse frequency
        return data.WeightedRandomSampler(
            weights=class_weights[self.y],
            num_samples=len(self),
            replacement=True
        )


def from_npz(ds, subset, transform=None):
    X, y = ds[f'X_{subset}'], ds[f'y_{subset}']
    return NumpyDataset(X, y, transform=transform)
