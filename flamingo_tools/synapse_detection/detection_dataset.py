from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import zarr

from scipy.spatial import cKDTree
from skimage.filters import gaussian
from skimage.feature import peak_local_max
from torch_em.util import ensure_tensor_with_channels

try:
    from spotiflow.utils import points_to_flow3d
    _spotiflow_available = True
except ImportError:
    _spotiflow_available = False


# Half width in voxels of the cube around an unannotated candidate that the loss ignores. It covers
# the response to a spot, a Gaussian of sigma 1 in the target.
IGNORE_RADIUS = 3
# A CTBP2 maximum this close to an annotation, in voxels, is the annotated synapse. This is the
# radius of the 2026-08-27 training-data diagnostics in to-do_revision/synapses.md.
CANDIDATE_MATCH_DISTANCE = 4


def read_csv_points(label_path: str) -> np.ndarray:
    """Read Napari CSV point annotations as an (N, 3) array in voxel coordinates, in ZYX order."""
    points_df = pd.read_csv(label_path)
    return np.stack([
        points_df["axis-0"].values, points_df["axis-1"].values, points_df["axis-2"].values,
    ], axis=1).astype(np.float32)


def find_unannotated_candidates(raw: np.ndarray, points: np.ndarray, percentile: float) -> np.ndarray:
    """Find the bright CTBP2 spots of a crop that no annotation explains.

    A local maximum of the smoothed image within `CANDIDATE_MATCH_DISTANCE` voxels of an
    annotation is matched. The candidates are the unmatched maxima that are at least as bright as
    `percentile` of the matched maxima. The loss must not supervise them as background, because
    their local profile resembles an annotated synapse.

    Args:
        raw: The CTBP2 image data of the full crop.
        points: The annotations of the crop in voxel coordinates, in the axis order of `raw`.
        percentile: The percentile of the matched maxima intensities that sets the threshold.

    Returns:
        The candidates in voxel coordinates. The array is empty when no maximum is matched,
        because the threshold cannot be calibrated then.
    """
    if raw.ndim != 3:
        raise ValueError(f"Expected a 3D crop, got shape {raw.shape}.")
    no_candidates = np.zeros((0, 3), dtype=np.float32)
    if len(points) == 0:
        return no_candidates

    smoothed = gaussian(raw.astype(np.float32), sigma=1)
    maxima = peak_local_max(smoothed, min_distance=2, exclude_border=False)
    if len(maxima) == 0:
        return no_candidates

    matched = cKDTree(points).query(maxima)[0] <= CANDIDATE_MATCH_DISTANCE
    if not matched.any():
        return no_candidates
    intensities = smoothed[tuple(maxima.T)]
    threshold = np.percentile(intensities[matched], percentile)
    return maxima[~matched & (intensities >= threshold)].astype(np.float32)


def _stamp_cubes(mask, points, radius, value):
    for point in np.round(points).astype(int):
        # Both ends are clamped, so that a point in front of the patch gives an empty slice
        # instead of a negative stop, which numpy would read as an offset from the end.
        mask[tuple(slice(max(0, coord - radius), max(0, coord + radius + 1)) for coord in point)] = value


class MinPointSampler:
    """A sampler to reject samples with too few foreground points.

    Args:
        min_points: The minimum number of points required to accept a sample.
        p_reject: The probability for rejecting a sample that does not meet the criterion.
    """
    def __init__(self, min_points: int, p_reject: float = 1.0):
        self.min_points = min_points
        self.p_reject = p_reject

    def __call__(self, x: np.ndarray, y: np.ndarray, rng=np.random) -> bool:
        """Check the sample.

        Args:
            x: The raw data.
            y: The label data as returned by the label transform (heatmap, or multi-channel
               heatmap+flow array with shape (C, Z, Y, X)).
            rng: The generator for the rejection draw. Pass a seeded generator to make the
                decision reproducible. The default uses the global numpy generator.

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
        mask_radius: Half width (in voxels) of the cube marked around each annotation in the loss
            mask. If given, the mask is appended as the last channel and the loss is restricted to
            it, so that unannotated CTBP2 spots outside the IHC region do not count as background.
            It also raises the halo to at least `mask_radius`.
        ignore_points: The unannotated candidates of each crop, keyed by the label path, from
            `find_unannotated_candidates`. If given, the loss mask excludes a cube of half width
            `IGNORE_RADIUS` around each candidate, but never the same cube around an annotation.
            Without `mask_radius` every other voxel stays in the mask.
    """

    # Context that DetectionDataset must load around each patch, before `mask_radius` raises it.
    # Only the flow and the loss mask need any.
    halo = 0

    def __init__(
        self,
        sigma: float,
        eps: float = 1e-8,
        mask_radius: Optional[int] = None,
        ignore_points: Optional[Dict[str, np.ndarray]] = None,
    ):
        self.sigma = sigma
        self.eps = eps
        self.mask_radius = mask_radius
        self.ignore_points = ignore_points
        # The mask must reach `mask_radius` beyond the patch, and the target has to be built from
        # the same annotations. Without the halo a cube around an annotation just outside the
        # patch would be masked in while its Gaussian is missing, which supervises a real synapse
        # as background. The same holds for the cube around a candidate. `self.halo` reads the
        # class attribute and shadows it per instance.
        self.halo = max(
            self.halo,
            0 if mask_radius is None else mask_radius,
            0 if ignore_points is None else IGNORE_RADIUS,
        )

    @staticmethod
    def _in_box(points, bb):
        """Return the points inside `bb` in box-local coordinates."""
        offset = np.array([s.start for s in bb], dtype=np.float32)
        inside = np.all(
            (points >= offset) & (points < np.array([s.stop for s in bb], dtype=np.float32)), axis=1
        )
        return points[inside] - offset

    @staticmethod
    def _local_points(label_path, bb):
        """Load the CSV points inside `bb` and return them in patch-local coordinates."""
        local_shape = tuple(s.stop - s.start for s in bb)
        return CsvHeatmapTransform._in_box(read_csv_points(label_path), bb), local_shape

    def _with_mask(self, labels, local_points, local_shape, label_path, bb):
        if self.mask_radius is None and self.ignore_points is None:
            return labels

        if self.mask_radius is None:
            mask = np.ones(local_shape, dtype=np.float32)
        else:
            mask = np.zeros(local_shape, dtype=np.float32)
            _stamp_cubes(mask, local_points, self.mask_radius, 1)

        if self.ignore_points is not None:
            _stamp_cubes(mask, self._in_box(self.ignore_points[label_path], bb), IGNORE_RADIUS, 0)
            # A candidate close to an annotation must not hide the annotated target.
            _stamp_cubes(mask, local_points, IGNORE_RADIUS, 1)

        return np.concatenate([labels, mask[np.newaxis]], axis=0)

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
        heatmap = self._heatmap(local_points, local_shape)
        return self._with_mask(heatmap[np.newaxis], local_points, local_shape, label_path, bb)


class CsvHeatmapFlowTransform(CsvHeatmapTransform):
    """Label transform for CSV point annotations that adds stereographic flow channels.

    The upstream czii-protein-challenge `HeatmapFlowTransform` reads JSON annotation files.
    This class reads the CSV files used by the local synapse training data and produces the same
    5-channel output (1 Gaussian heatmap + 4 stereographic flow channels).

    Args:
        sigma: Gaussian standard deviation (in voxels) for the heatmap.
        eps: Small constant added when normalizing the heatmap.
        mask_radius: Half width (in voxels) of the cube marked around each annotation in the loss
            mask. If given, the mask is appended as a sixth channel.
        ignore_points: The unannotated candidates of each crop, see `CsvHeatmapTransform`.
    """

    # The flow needs points beyond the patch border to be correct close to the border.
    halo = 10

    def __init__(
        self,
        sigma: float,
        eps: float = 1e-8,
        mask_radius: Optional[int] = None,
        ignore_points: Optional[Dict[str, np.ndarray]] = None,
    ):
        if not _spotiflow_available:
            raise ImportError(
                "spotiflow is required for flow computation. "
                'Install it with: pip install "cochlea_net[flow]"'
            )
        super().__init__(sigma, eps, mask_radius, ignore_points)

    def __call__(self, label_path, shape, bb_labels, bb_for_loading):
        bb = bb_for_loading[-3:] if len(bb_for_loading) > 3 else bb_for_loading
        local_points, local_shape = self._local_points(label_path, bb)

        heatmap = self._heatmap(local_points, local_shape)
        if len(local_points) == 0:
            flow = np.zeros((4, *local_shape), dtype=np.float32)
        else:
            flow = points_to_flow3d(local_points, local_shape)  # returns (Z', Y', X', 4)
            flow = np.asarray(flow, dtype=np.float32).transpose((3, 0, 1, 2))  # -> (4, Z', Y', X')

        labels = np.concatenate([heatmap[np.newaxis], flow], axis=0).astype(np.float32)
        return self._with_mask(labels, local_points, local_shape, label_path, bb)


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
        patch_seed=None,
    ):
        self.raw_path = raw_path
        self.label_path = label_path
        self.raw_key = raw_key
        self._ndim = 3

        assert len(patch_shape) == self._ndim
        self.patch_shape = patch_shape

        self.raw_transform = raw_transform
        # `sigma` and `eps` only feed this fallback, for a dataset constructed without a transform.
        if label_transform is None:
            label_transform = CsvHeatmapTransform(sigma, eps)
        self.label_transform = label_transform
        self.label_transform2 = label_transform2
        self.transform = transform
        self.sampler = sampler
        # Makes the patch for a given index reproducible, for the validation set. Leave it unset
        # for training, where every access should see a new patch.
        self.patch_seed = patch_seed

        self.dtype = dtype
        self.label_dtype = label_dtype

        # Buffer added around each sampled patch before calling the label transform. The label
        # transform declares how much context it needs: the flow and the loss mask.
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

        if self.raw_transform is not None:
            raw = self.raw_transform(raw)

        if self.transform is not None:
            raw, labels = self.transform(raw, labels)

        if self.label_transform2 is not None:
            labels = self.label_transform2(labels)

        raw = ensure_tensor_with_channels(raw, ndim=self._ndim, dtype=self.dtype)
        labels = ensure_tensor_with_channels(labels, ndim=self._ndim, dtype=self.label_dtype)
        return raw, labels
