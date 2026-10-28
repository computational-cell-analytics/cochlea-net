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


def read_csv_points(label_path: str) -> np.ndarray:
    """Read Napari CSV point annotations as an (N, 3) array in voxel coordinates, in ZYX order.

    A consensus annotation also lists the points that only one annotator marked, in an `annotator`
    column. Only the `consensus` rows are returned for such a file.
    """
    points_df = pd.read_csv(label_path)
    if "annotator" in points_df.columns:
        points_df = points_df[points_df["annotator"] == "consensus"]
    return np.stack([
        points_df["axis-0"].values, points_df["axis-1"].values, points_df["axis-2"].values,
    ], axis=1).astype(np.float32)


class MinPointSampler:
    """A sampler to reject samples with too few foreground points.

    Args:
        min_points: The minimum number of points required to accept a sample.
        p_reject: The probability for rejecting a sample that does not meet the criterion.
    """
    def __init__(self, min_points: int, p_reject: float = 1.0):
        self.min_points = min_points
        self.p_reject = p_reject

    def __call__(self, x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> bool:
        """Check the sample.

        Args:
            x: The raw data.
            y: The label data as returned by the label transform (heatmap, or multi-channel
               heatmap+flow array with shape (C, Z, Y, X)).
            rng: The generator for the rejection draw. Pass a seeded generator to make the
                decision reproducible.

        Returns:
            Whether to accept this sample.
        """
        heatmap = y[0] if y.ndim == 4 else y
        n_points = len(peak_local_max(heatmap, min_distance=2, threshold_rel=0.3))
        if n_points > self.min_points:
            return True
        return rng.random() > self.p_reject


class CsvHeatmapTransform:
    """Label transform for CSV point annotations that produces a Gaussian heatmap.

    The class matches the `(label_path, shape, bb_labels, bb_for_loading)` loader interface of the
    upstream czii-protein-challenge `HeatmapTransform`, but reads the Napari CSV files used by the
    local synapse training data. The single output channel is the target of the v3 model.

    Args:
        sigma: Gaussian standard deviation (in voxels) for the heatmap.
        eps: Small constant added when normalizing the heatmap.
        fixed_target: Give every isolated annotation a peak of 4, and load the annotations up to
            3 sigma beyond the patch. Without it, the target of every model up to v10 is
            reproduced: the heatmap is divided by its patch maximum, and an annotation on the
            border of the loaded box raises that maximum, which lowers every other peak in the
            patch to 2.28. An annotation just outside the patch also loses its Gaussian.
    """

    # Context that DetectionDataset must load around each patch. Only the flow and the fixed
    # target need any.
    halo = 0

    def __init__(self, sigma: float, eps: float = 1e-8, fixed_target: bool = False):
        self.sigma = sigma
        self.eps = eps
        self.fixed_target = fixed_target
        if fixed_target:
            # `self.halo` reads the class attribute and shadows it per instance.
            self.halo = max(self.halo, int(np.ceil(3 * sigma)))
            radius = int(np.ceil(4 * sigma))
            delta = np.zeros((2 * radius + 1,) * 3, dtype=np.float32)
            delta[radius, radius, radius] = 1
            self._peak = float(gaussian(delta, sigma, mode="constant").max())

    @staticmethod
    def _local_points(label_path, bb):
        """Load the CSV points inside `bb` and return them in patch-local coordinates."""
        local_shape = tuple(s.stop - s.start for s in bb)
        points = read_csv_points(label_path)
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
            if self.fixed_target:
                heatmap = gaussian(heatmap, self.sigma, mode="constant") * (4 / self._peak)
            else:
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
        fixed_target: See `CsvHeatmapTransform`.
    """

    # The flow needs points beyond the patch border to be correct close to the border.
    halo = 10

    def __init__(self, sigma: float, eps: float = 1e-8, fixed_target: bool = False):
        if not _spotiflow_available:
            raise ImportError(
                "spotiflow is required for flow computation. "
                'Install it with: pip install ".[flow]" in the cochlea-net repository.'
            )
        super().__init__(sigma, eps, fixed_target)

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

    def __init__(
        self,
        raw_path,
        raw_key,
        label_path,
        patch_shape,
        label_transform,
        n_samples=None,
        sampler=None,
        patch_seed=None,
    ):
        self.raw_path = raw_path
        self.label_path = label_path
        self.raw_key = raw_key
        self._ndim = 3

        assert len(patch_shape) == self._ndim
        self.patch_shape = patch_shape

        self.label_transform = label_transform
        self.sampler = sampler
        # Makes the patch for a given index reproducible, for the validation set. Leave it unset
        # for training, where every access should see a new patch.
        self.patch_seed = patch_seed

        # Buffer added around each sampled patch before calling the label transform. The label
        # transform declares how much context it needs: the flow and the fixed target.
        self.halo = getattr(label_transform, "halo", 10)

        f = zarr.open(self.raw_path, mode="r")
        full_shape = f[self.raw_key].shape

        # Determine 3D spatial shape, stripping an optional channel dim.
        if len(full_shape) == 4:
            self.shape = full_shape[:-1] if full_shape[-1] < 16 else full_shape[1:]
        else:
            self.shape = full_shape

        if n_samples is None:
            n_samples = int(np.prod([sh / psh for sh, psh in zip(self.shape, self.patch_shape)]))
        self._len = n_samples

    def __len__(self):
        return self._len

    @property
    def ndim(self):
        return self._ndim

    def _sample_bounding_box(self, rng):
        if any(sh < psh for sh, psh in zip(self.shape, self.patch_shape)):
            raise NotImplementedError(
                f"Image padding is not supported yet. Data shape {self.shape}, patch shape {self.patch_shape}"
            )
        # The halo needs no margin here: `_get_desired_raw_and_labels` clamps it to the volume.
        bb_start = [rng.integers(0, sh - psh + 1) for sh, psh in zip(self.shape, self.patch_shape)]
        return tuple(slice(start, start + psh) for start, psh in zip(bb_start, self.patch_shape))

    def _get_desired_raw_and_labels(self, rng):
        raw = zarr.open(self.raw_path, mode="r")[self.raw_key]
        have_raw_channels = raw.ndim == 4

        bb = self._sample_bounding_box(rng)

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

        # The label transform is the label loader. It receives the path and bounding box and
        # returns an array covering bb_for_loading; we then crop the halo back out.
        labels = self.label_transform(self.label_path, self.shape, bb_for_loading, bb_for_loading)
        if labels.ndim == 4:
            labels = labels[(slice(None),) + slices_crop]
        else:
            labels = labels[slices_crop]

        return raw_patch, labels

    def _get_sample(self, index):
        # With a patch seed the patch is a pure function of (patch_seed, index), so the same
        # index gives the same patch in every epoch and in every data loader worker. Without one
        # every access draws a new patch, which is what training wants.
        rng = np.random.default_rng(None if self.patch_seed is None else (self.patch_seed, index))
        raw, labels = self._get_desired_raw_and_labels(rng)

        if self.sampler is not None:
            sample_id = 0
            while not self.sampler(raw, labels, rng):
                raw, labels = self._get_desired_raw_and_labels(rng)
                sample_id += 1
                if sample_id > self.max_sampling_attempts:
                    raise RuntimeError(
                        f"Could not sample a valid batch in {self.max_sampling_attempts} attempts"
                    )

        return raw, labels

    def __getitem__(self, index):
        raw, labels = self._get_sample(index)
        raw = ensure_tensor_with_channels(raw, ndim=self._ndim, dtype=torch.float32)
        labels = ensure_tensor_with_channels(labels, ndim=self._ndim, dtype=torch.float32)
        return raw, labels
