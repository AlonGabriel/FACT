import numpy as np
import scipy.interpolate as interpolate
import torch
import torchvision.transforms as T


class IntensityAwareAugmentation:

    def __init__(self, ticks_range=(100, 1000), tick_shift_limit=2, signature_threshold=0.15,
                 background_noise_ratio=0.7, background_noise_spline_points=4,
                 signature_noise_ratio=0.3,
                 random_state=None):
        """
        Intensity-aware mass spectrum augmentation.

        Adapted and refactored from the original implementation by @moon (2022).

        Parameters
        ----------
        ticks_range: tuple of int
            Start and end of the m/z range.
        tick_shift_limit: int
            Maximum shift (noise) in m/z ticks.
        signature_threshold: float
            Peaks higher than this threshold are considered signature peaks, otherwise background. Signature peaks are
            augmented with higher intensity noise, while background peaks are augmented with lower intensity noise.
        background_noise_ratio: float
            Ratio of the STD of the background noise to the STD of the spectrum.
        background_noise_spline_points: int
            Number of spline points for generating background noise.
        signature_noise_ratio: float
            Ratio of the STD of the signature noise to the STD of the spectrum.
        random_state: int
            Random seed for reproducibility.
        """
        self.ticks_range = ticks_range
        self.tick_shift_limit = tick_shift_limit
        self.signature_threshold = signature_threshold
        self.background_noise_ratio = background_noise_ratio
        self.background_noise_spline_points = background_noise_spline_points
        self.signature_noise_ratio = signature_noise_ratio
        self.rng = np.random.default_rng(random_state)
        self.gen = torch.Generator()
        if random_state:
            self.gen.manual_seed(random_state)

    def __call__(self, spectra):
        """
        Augments a batch of mass spectra.

        Parameters
        ----------
        spectra: torch.Tensor
            A batch of mass spectra, shape (num_spectra, num_bins). The spectra are assumed to be normalized.

        Returns
        -------
        torch.Tensor
            The augmented batch of mass spectra.
        """
        shifted_indices = self.shifted_indices(spectra)
        background_noise = self.background_noise(spectra)
        signature_noise = self.signature_noise(spectra)

        augmented = spectra + torch.where(spectra > self.signature_threshold, signature_noise, background_noise)
        augmented = [tensor[indices] for tensor, indices in zip(augmented, shifted_indices)]
        augmented = torch.stack(augmented, dim=0)
        augmented = torch.abs(augmented)

        return augmented

    def background_noise(self, spectra):
        num_spectra, num_bins = spectra.shape
        start_tick, end_tick = self.ticks_range
        ticks = torch.arange(*self.ticks_range)

        strengths = torch.rand(num_spectra, 1, generator=self.gen).to(spectra) * torch.std(spectra, dim=1, keepdim=True) * self.background_noise_ratio
        amplitudes = strengths * torch.randn(num_spectra, num_bins, generator=self.gen).to(spectra)

        # We'll do the next part in Numpy since we want to use `interpolate.splrep` and `interpolate.splev`.
        spline_middle_ticks = [self.rng.choice(ticks[2:-2], size=self.background_noise_spline_points, replace=False) for i in range(num_spectra)]
        spline_xs = [[start_tick, *sorted(ticks), end_tick] for ticks in spline_middle_ticks]
        spline_ys = self.rng.random((num_spectra, self.background_noise_spline_points + 2))
        knots = [interpolate.splrep(xs, ys) for xs, ys in zip(spline_xs, spline_ys)]
        noise = [interpolate.splev(ticks, el) for el in knots]
        noise = np.array(noise)

        minimum = noise.min(axis=1, keepdims=True)
        maximum = noise.max(axis=1, keepdims=True)
        noise = (noise - minimum) / (maximum - minimum)
        noise = torch.from_numpy(noise).to(spectra)

        return amplitudes * noise

    def signature_noise(self, spectra):
        num_spectra, num_bins = spectra.shape

        strengths = torch.rand(num_spectra, 1, generator=self.gen).to(spectra) * torch.std(spectra, dim=1, keepdim=True) * self.signature_noise_ratio
        amplitudes = strengths * torch.randn(num_spectra, num_bins, generator=self.gen).to(spectra)

        return amplitudes

    def shifted_indices(self, spectra):
        num_spectra, num_bins = spectra.shape
        ticks = np.arange(*self.ticks_range)

        strengths = self.rng.random(size=(num_spectra, 1))
        offsets = np.round(strengths * self.rng.normal(size=spectra.shape))
        offsets = np.clip(offsets, -self.tick_shift_limit, self.tick_shift_limit)

        augmented_ticks = ticks + offsets
        augmented_ticks = np.argsort(augmented_ticks, axis=1)

        return torch.from_numpy(augmented_ticks).to(spectra.device)


class RandomImSpectAugmentation:

    def __init__(self, resized_crop=True, color_distortion=True, gaussian_blur=True, im_size=224, distortion=1):
        """
        Applies a series of image augmentation techniques to a batch of images.

        Adapted from https://github.com/AndrewAtanov/simclr-pytorch/blob/master/models/ssl.py.

        Parameters
        ----------
        resized_crop: bool
            Whether to apply resized crop.
        color_distortion: bool
            Whether to apply color distortion.
        gaussian_blur: bool
            Whether to apply Gaussian blur.
        im_size: int
            Image size.
        distortion: float
            Color distortion strength.
        """
        self.resized_crop = None
        if resized_crop:
            self.resized_crop = T.RandomResizedCrop(im_size, (0.08, 1), interpolation=T.InterpolationMode.BICUBIC)
        self.color_distortion = None
        if color_distortion:
            color_jitter = T.ColorJitter(0.8 * distortion, 0.8 * distortion, 0.8 * distortion, 0.2 * distortion)
            self.color_distortion = T.Compose([
                T.RandomApply([color_jitter], p=0.8),
                T.RandomGrayscale(p=0.2)
            ])
        self.gaussian_blur = None
        if gaussian_blur:
            self.gaussian_blur = T.GaussianBlur(np.ceil(im_size / 10), 0.5)

    def __call__(self, images):
        if self.resized_crop:
            images = self.apply(self.resized_crop, images)
        if self.color_distortion:
            images = self.apply(self.color_distortion, images)
        if self.gaussian_blur:
            images = self.apply(self.gaussian_blur, images)
        return images

    @classmethod
    def apply(cls, func, batch):
        return torch.stack([func(el) for el in batch])


construct = {
    'laion/clap-htsat-unfused': IntensityAwareAugmentation,
    'openai/clip-vit-base-patch32': RandomImSpectAugmentation,
    'pluskal-lab/DreaMS': IntensityAwareAugmentation,
}


def construct_augmenter(base_model):
    return construct[base_model]()
