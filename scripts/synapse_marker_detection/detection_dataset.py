import numpy as np
import pandas as pd
import torch
import zarr

from skimage.filters import gaussian
from skimage.feature import peak_local_max
from torch_em.transform.raw import standardize
from torch_em.util import ensure_tensor_with_channels


class MinPointSampler:
    """A sampler to reject samples with too few foreground points.

    Args:
        min_points: The minimum number of points required to accept a sample.
        p_reject: The probability for rejecting a sample that does not meet the criterion.
    """
    def __init__(self, min_points: int, p_reject: float = 1.0):
        self.min_points = min_points
        self.p_reject = p_reject

    def __call__(self, x: np.ndarray, y: np.ndarray) -> bool:
        """Check the sample.

        Args:
            x: The raw data.
            y: The label data as returned by the label transform (heatmap, or multi-channel
               heatmap+flow array with shape (C, Z, Y, X)).

        Returns:
            Whether to accept this sample.
        """
        heatmap = y[0] if y.ndim == 4 else y
        n_points = len(peak_local_max(heatmap, min_distance=2, threshold_rel=0.3))
        if n_points > self.min_points:
            return True
        return np.random.rand() > self.p_reject


def load_labels(label_path, shape, bb):
    """Load point labels from a CSV file, optionally restricted to a bounding box."""
    points = pd.read_csv(label_path)
    assert len(points.columns) == len(shape)
    z_coords, y_coords, x_coords = points["axis-0"].values, points["axis-1"].values, points["axis-2"].values

    if bb is not None:
        (z_min, z_max), (y_min, y_max), (x_min, x_max) = [(s.start, s.stop) for s in bb]
        z_coords -= z_min
        y_coords -= y_min
        x_coords -= x_min
        mask = np.logical_and.reduce([
            np.logical_and(z_coords >= 0, z_coords < (z_max - z_min)),
            np.logical_and(y_coords >= 0, y_coords < (y_max - y_min)),
            np.logical_and(x_coords >= 0, x_coords < (x_max - x_min)),
        ])
        z_coords, y_coords, x_coords = z_coords[mask], y_coords[mask], x_coords[mask]
        shape = (z_max - z_min, y_max - y_min, x_max - x_min)

    n_points = len(z_coords)
    coords = tuple(
        np.clip(np.round(coord).astype("int"), 0, coord_max - 1) for coord, coord_max in zip(
            (z_coords, y_coords, x_coords), shape
        )
    )
    return coords, n_points


def process_labels(coords, shape, sigma, eps, bb=None):
    """Create a normalized Gaussian heatmap from point coordinates."""
    if bb:
        (z_min, z_max), (y_min, y_max), (x_min, x_max) = [(s.start, s.stop) for s in bb]
        shape = (z_max - z_min, y_max - y_min, x_max - x_min)

    labels = np.zeros(shape, dtype="float32")
    labels[coords] = 1
    labels = gaussian(labels, sigma)
    labels /= (labels.max() + 1e-7)
    labels *= 4
    return labels


class DetectionDataset(torch.utils.data.Dataset):
    max_sampling_attempts = 500

    @staticmethod
    def compute_len(shape, patch_shape):
        if patch_shape is None:
            return 1
        else:
            n_samples = int(np.prod([float(sh / csh) for sh, csh in zip(shape, patch_shape)]))
            return n_samples

    def __init__(
        self,
        raw_path,
        raw_key,
        label_path,
        patch_shape,
        raw_transform=None,
        label_transform=None,
        label_transform2=None,
        transform=None,
        dtype=torch.float32,
        label_dtype=torch.float32,
        n_samples=None,
        sampler=None,
        eps=1e-8,
        sigma=None,
        lower_bound=None,
        upper_bound=None,
        **kwargs,
    ):
        self.raw_path = raw_path
        self.label_path = label_path
        self.raw_key = raw_key
        self._ndim = 3

        assert len(patch_shape) == self._ndim
        self.patch_shape = patch_shape

        self.raw_transform = raw_transform
        self.label_transform = label_transform
        self.label_transform2 = label_transform2
        self.transform = transform
        self.sampler = sampler

        self.dtype = dtype
        self.label_dtype = label_dtype

        self.eps = eps
        self.sigma = sigma
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        # Buffer added around each sampled patch before calling the label transform,
        # so that HeatmapFlowTransform has context to compute flow near patch edges.
        self.halo = 10

        with zarr.open(self.raw_path, "r") as f:
            full_shape = f[self.raw_key].shape

        # Determine 3D spatial shape, stripping an optional channel dim.
        if len(full_shape) == 4:
            self.shape = full_shape[:-1] if full_shape[-1] < 16 else full_shape[1:]
        else:
            self.shape = full_shape

        self._len = self.compute_len(self.shape, self.patch_shape) if n_samples is None else n_samples

    def __len__(self):
        return self._len

    @property
    def ndim(self):
        return self._ndim

    def _sample_bounding_box(self):
        if any(sh < psh for sh, psh in zip(self.shape, self.patch_shape)):
            raise NotImplementedError(
                f"Image padding is not supported yet. Data shape {self.shape}, patch shape {self.patch_shape}"
            )
        bb_start = [
            np.random.randint(0, max(1, sh - psh - 2 * self.halo))
            for sh, psh in zip(self.shape, self.patch_shape)
        ]
        return tuple(slice(start, start + psh) for start, psh in zip(bb_start, self.patch_shape))

    def _get_desired_raw_and_labels(self):
        raw = zarr.open(self.raw_path, "r")[self.raw_key]
        have_raw_channels = raw.ndim == 4

        bb = self._sample_bounding_box()

        # Extend the patch bounding box with halo on each side, clamped to the volume.
        bb_for_loading = tuple(
            slice(max(0, s.start - self.halo), min(self.shape[i], s.stop + self.halo))
            for i, s in enumerate(bb)
        )

        # Load raw with channel handling.
        prefix_box = tuple()
        if have_raw_channels and raw.shape[-1] >= 16:
            # channels-first layout: prepend slice(None) to select all channels
            prefix_box = (slice(None),)

        raw_patch = np.array(raw[prefix_box + bb_for_loading])

        # Compute crop slices that remove the halo and restore exactly patch_shape.
        slices_crop = tuple(
            slice(s.start - bl.start, s.start - bl.start + psh)
            for s, bl, psh in zip(bb, bb_for_loading, self.patch_shape)
        )

        if have_raw_channels and len(prefix_box) == 0:
            # channels-last layout: (Z, Y, X, C) → crop → (C, Z, Y, X)
            raw_patch = raw_patch[slices_crop + (slice(None),)].transpose((3, 0, 1, 2))
        elif have_raw_channels:
            raw_patch = raw_patch[(slice(None),) + slices_crop]
        else:
            raw_patch = raw_patch[slices_crop]

        # Generate labels.
        if self.label_transform is not None:
            # label_transform is the label loader (e.g. HeatmapFlowTransform from the upstream
            # czii-protein-challenge repo). It receives the path and bounding box and returns
            # a (C, Z, Y, X) array covering bb_for_loading; we then crop the halo back out.
            labels = self.label_transform(self.label_path, self.shape, bb_for_loading, bb_for_loading)
            if labels.ndim == 4:
                labels = labels[(slice(None),) + slices_crop]
            else:
                labels = labels[slices_crop]
        else:
            # Fallback: load CSV point coordinates and build a single-channel Gaussian heatmap.
            coords, _ = load_labels(self.label_path, self.shape, bb)
            labels = process_labels(coords, self.shape, self.sigma, self.eps, bb=bb)

        return raw_patch, labels

    def _get_sample(self, index):
        raw, labels = self._get_desired_raw_and_labels()

        if self.sampler is not None:
            sample_id = 0
            while not self.sampler(raw, labels):
                raw, labels = self._get_desired_raw_and_labels()
                sample_id += 1
                if sample_id > self.max_sampling_attempts:
                    raise RuntimeError(
                        f"Could not sample a valid batch in {self.max_sampling_attempts} attempts"
                    )

        return raw, labels

    def __getitem__(self, index):
        raw, labels = self._get_sample(index)

        if self.raw_transform is not None:
            raw = self.raw_transform(raw)

        if self.transform is not None:
            raw, labels = self.transform(raw, labels)

        if self.label_transform2 is not None:
            labels = self.label_transform2(labels)

        raw = ensure_tensor_with_channels(raw, ndim=self._ndim, dtype=self.dtype)
        labels = ensure_tensor_with_channels(labels, ndim=self._ndim, dtype=self.label_dtype)
        return raw, labels


if __name__ == "__main__":
    import napari

    raw_path = "training_data/images/10.1L_mid_IHCribboncount_5_Z.zarr"
    label_path = "training_data/labels/10.1L_mid_IHCribboncount_5_Z.csv"

    f = zarr.open(raw_path, "r")
    raw = f["raw"][:]

    coords, _ = load_labels(label_path, raw.shape, bb=None)
    labels = process_labels(coords, shape=raw.shape, sigma=1, eps=1e-7)

    v = napari.Viewer()
    v.add_image(raw)
    v.add_image(labels)
    napari.run()
