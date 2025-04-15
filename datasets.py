import numpy as np
import pathlib
import torch.utils.data as data
import torch
from torch.utils.data import WeightedRandomSampler



class NumpyDataset(data.Dataset):

    def __init__(self, X, y, transform=None, output_labels=True, fit_transform=False):
        if transform:
            if fit_transform and hasattr(transform, "fit_transform"):
                self.X = transform.fit_transform(X)  # Fit and transform if needed
            elif hasattr(transform, "transform"):
                self.X = transform.transform(X)  # Just transform
            else:
                raise ValueError("Transform must implement either fit_transform or transform methods.")
        else:
            self.X = X
        self.y = y
        self.output_labels = output_labels

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        x = self.X[index].astype(np.float32)
        if not self.output_labels:
            return x
        y = self.y[index]
        y = y.astype(np.int64)
        return x, y

    def weighted_sampler(self):
        labels, counts = np.unique(self.y, return_counts=True)
        print(f"Class Counts: {dict(zip(labels, counts))}")  # ✅ Print class counts

        class_weights = 1.0 / torch.tensor(counts, dtype=torch.float)  # ✅ Compute inverse frequency
        print(f"Computed Class Weights: {class_weights}")  # ✅ Print class weights

        # 🔥 Convert self.y from numpy to PyTorch long tensor
        y_tensor = torch.tensor(self.y, dtype=torch.long)

        # 🔥 Assign weights using proper indexing
        sample_weights = class_weights[y_tensor]


        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(self),
            replacement=True
        )

def from_npz(ds, subset, transform=None, scaler_path=None):
    X, y = ds[f'X_{subset}'], ds[f'y_{subset}']
    # Ensure data is shuffled at dataset loading
    indices = np.arange(len(y))
    np.random.shuffle(indices)  # Shuffle indices
    X, y = X[indices], y[indices]  # Apply shuffle
    # Determine whether to fit or just transform
    if subset == 'train' and transform and scaler_path:
        # Check if the scaler already exists
        if pathlib.Path(scaler_path).exists():
            transform.load_scaler(scaler_path)
            fit_transform = False
        else:
            fit_transform = True
    else:
        fit_transform = False

    return NumpyDataset(X, y, transform=transform, fit_transform=fit_transform)


class ZippedLoader:

    def __init__(self, primary, secondary):
        self.primary = primary
        self.secondary = secondary

    def __len__(self):
        return len(self.primary)

    def __iter__(self):
        secondary = iter(self.secondary)
        for batch in self.primary:
            secondary_batch = next(secondary)
            print(f"Primary batch: {type(batch)}, Secondary batch: {type(secondary_batch)}")
            yield batch, next(secondary)

