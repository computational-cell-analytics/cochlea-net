import numpy as np
import pandas as pd
import torch
import zarr

from skimage.filters import gaussian
from skimage.feature import peak_local_max
from torch_em.util import ensure_tensor_with_channels

try:
    from spotiflow.utils import points_to_flow3d
    _spotiflow_available = True
except ImportError:
    _spotiflow_available = False


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


class CsvHeatmapTransform:
    """Label transform for CSV point annotations that produces a Gaussian heatmap.

    The class matches the `(label_path, shape, bb_labels, bb_for_loading)` loader interface of the
    upstream czii-protein-challenge `HeatmapTransform`, but reads the Napari CSV files used by the
    local synapse training data. The single output channel is the target of the v3 model.

    Args:
        sigma: Gaussian standard deviation (in voxels) for the heatmap.
        eps: Small constant added when normalizing the heatmap.
    """

    # Context that DetectionDataset must load around each patch. Only the flow needs any.
    halo = 0

    def __init__(self, sigma: float, eps: float = 1e-8):
        self.sigma = sigma
        self.eps = eps

    @staticmethod
    def _local_points(label_path, bb):
        """Load the CSV points inside `bb` and return them in patch-local coordinates."""
        local_shape = tuple(s.stop - s.start for s in bb)
        points_df = pd.read_csv(label_path)
        points = np.stack([
            points_df["axis-0"].values, points_df["axis-1"].values, points_df["axis-2"].values,
        ], axis=1).astype(np.float32)

        offset = np.array([s.start for s in bb], dtype=np.float32)
        mask = np.all(
            (points >= offset) & (points < np.array([s.stop for s in bb], dtype=np.float32)), axis=1
        )
        return points[mask] - offset, local_shape

    def _heatmap(self, local_points, local_shape):
        heatmap = np.zeros(local_shape, dtype=np.float32)
        if len(local_points) > 0:
            coords = tuple(
                np.clip(np.round(coord).astype(int), 0, size - 1)
                for coord, size in zip(local_points.T, local_shape)
            )
            heatmap[coords] = 1
            heatmap = gaussian(heatmap, self.sigma)
            heatmap /= (heatmap.max() + self.eps)
            heatmap *= 4
        return heatmap

    def __call__(self, label_path, shape, bb_labels, bb_for_loading):
        # Strip a leading channel slice if present (bb_for_loading may have one).
        bb = bb_for_loading[-3:] if len(bb_for_loading) > 3 else bb_for_loading
        local_points, local_shape = self._local_points(label_path, bb)
        return self._heatmap(local_points, local_shape)[np.newaxis]


class CsvHeatmapFlowTransform(CsvHeatmapTransform):
    """Label transform for CSV point annotations that adds stereographic flow channels.

    The upstream czii-protein-challenge `HeatmapFlowTransform` reads JSON annotation files.
    This class reads the CSV files used by the local synapse training data and produces the same
    5-channel output (1 Gaussian heatmap + 4 stereographic flow channels).

    Args:
        sigma: Gaussian standard deviation (in voxels) for the heatmap.
        eps: Small constant added when normalizing the heatmap.
    """

    # The flow needs points beyond the patch border to be correct close to the border.
    halo = 10

    def __init__(self, sigma: float, eps: float = 1e-8):
        if not _spotiflow_available:
            raise ImportError(
                "spotiflow is required for flow computation. "
                "Install it with: pip install spotiflow"
            )
        super().__init__(sigma, eps)

    def __call__(self, label_path, shape, bb_labels, bb_for_loading):
        bb = bb_for_loading[-3:] if len(bb_for_loading) > 3 else bb_for_loading
        local_points, local_shape = self._local_points(label_path, bb)

        heatmap = self._heatmap(local_points, local_shape)
        if len(local_points) == 0:
            flow = np.zeros((4, *local_shape), dtype=np.float32)
        else:
            flow = points_to_flow3d(local_points, local_shape)  # returns (Z', Y', X', 4)
            flow = np.asarray(flow, dtype=np.float32).transpose((3, 0, 1, 2))  # -> (4, Z', Y', X')

        return np.concatenate([heatmap[np.newaxis], flow], axis=0).astype(np.float32)


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
        # The upstream loader always supplies a label transform. Fall back to the plain heatmap
        # target when the dataset is constructed directly.
        if label_transform is None:
            label_transform = CsvHeatmapTransform(sigma, eps)
        self.label_transform = label_transform
        self.label_transform2 = label_transform2
        self.transform = transform
        self.sampler = sampler

        self.dtype = dtype
        self.label_dtype = label_dtype

        # Accepted because the upstream loader always passes them; the label transform owns them.
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        # Buffer added around each sampled patch before calling the label transform. The label
        # transform declares how much context it needs; only the flow computation needs any.
        self.halo = getattr(label_transform, "halo", 10)

        f = zarr.open(self.raw_path, mode="r")
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
        raw = zarr.open(self.raw_path, mode="r")[self.raw_key]
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

        # The label transform is the label loader (e.g. HeatmapFlowTransform from the upstream
        # czii-protein-challenge repo). It receives the path and bounding box and returns an array
        # covering bb_for_loading; we then crop the halo back out. The upstream transforms return
        # (Z, Y, X) for a plain heatmap and (C, Z, Y, X) with flow.
        labels = self.label_transform(self.label_path, self.shape, bb_for_loading, bb_for_loading)
        if labels.ndim == 4:
            labels = labels[(slice(None),) + slices_crop]
        else:
            labels = labels[slices_crop]

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
