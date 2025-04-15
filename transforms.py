import numpy as np
from sklearn.preprocessing import MinMaxScaler
from joblib import dump, load
import pathlib


class Normalize:

    def __init__(self, mean, std):
        self.mean = np.asarray(mean)
        self.std = np.asarray(std)

    def __call__(self, array):
        return (array - self.B(self.mean)) / self.B(self.std)

    @classmethod
    def B(cls, stat):  # Broadcast
        return stat[:, np.newaxis, np.newaxis]


class MinMaxNormalize:
    def __init__(self, scaler_path=None):
        self.scaler = MinMaxScaler()
        self.is_fitted = False  # Track if the scaler has been fit
        self.scaler_path = scaler_path  # Optional path for saving/loading the scaler

    def fit_transform(self, array):
        self.is_fitted = True
        print("Fitting and transforming data")
        norm_array = self.scaler.fit_transform(array)
        norm_array = norm_array.clip(0, 1)
        if self.scaler_path:
            self.save_scaler()
        return norm_array

    def transform(self, array):
        if not self.is_fitted:
            raise RuntimeError("Scaler has not been fitted. Call fit_transform on training data first.")
        print("Transforming data")
        norm_array = self.scaler.transform(array)
        norm_array = norm_array.clip(0, 1)
        return norm_array

    def __call__(self, array, is_training=False):
        if is_training and not self.is_fitted:
            return self.fit_transform(array)
        elif not is_training:
            return self.transform(array)
        else:
            raise RuntimeError("Scaler is already fitted. Avoid calling fit_transform multiple times.")

    def save_scaler(self):
        if self.scaler_path:
            pathlib.Path(self.scaler_path).parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
            dump(self.scaler, self.scaler_path)
           
    def load_scaler(self, path):
        self.scaler = load(path)
        self.is_fitted = True
        

class SampleNormalize:
    """
    Perform sample-based normalization (spectrum-based) on the input array.
    Normalizes each sample (spectrum) independently so that the range of ion intensities in each
    spectrum becomes 0–1.
    """
    def transform(self, array):
        """
        Normalize each spectrum (row) independently.
        
        Args:
            array (np.ndarray): Input array of shape (N, P), where N is the number of spectra and
                                P is the number of ion features.
        
        Returns:
            np.ndarray: Spectrum-normalized array with the same shape as input.
        """
        # Compute row-wise min and max
        row_min = array.min(axis=1, keepdims=True)  # Shape: (N, 1)
        row_max = array.max(axis=1, keepdims=True)  # Shape: (N, 1)
        row_range = row_max - row_min

        # Avoid division by zero
        row_range[row_range == 0] = 1

        # Normalize each spectrum independently
        normalized_array = (array - row_min) / row_range

        return normalized_array
