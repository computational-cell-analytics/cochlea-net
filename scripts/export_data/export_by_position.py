"""Export 2D crops at standard tonotopic positions (default: apex/mid/base) for a single
cochlea, dispatching to one or more of the export_data scripts in this folder.

The script has two modes:
- `--groups` expands named export groups into the export passes they need, e.g. "sgn" for
  PV plus the SGN segmentation and "ihc" for VGlut3, CTBP2, the IHC segmentation and the
  synapse detections. Every group resolves its crop centers from its own reference
  segmentation, so an SGN gallery is centered on the SGN table and an IHC gallery on the IHC
  table within the same run. Each group writes to its own sub-folder of the output folder.
- `--export_functions` / `--json_info` dispatch export functions directly, with one shared
  reference segmentation. This is the route for the marker, subtypes and frequency exports.

Crop centers are resolved from a reference segmentation table (the same lookup that powers the
flamingo_tools.table_info command) and are reused unchanged across every export pass that
shares the reference, since crop centers are physical µm coordinates shared across all
co-registered channels of a cochlea. With `--view_only` the crop centers come from the exported
file names instead, so a finished export can be reviewed without any S3 access.
"""
import argparse
import json
import os
from itertools import cycle
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import tifffile

import export_frequency_mapping
import export_lower_resolution
import export_lower_resolution_marker
import export_lower_resolution_subtypes
import export_synapse_detections

from flamingo_tools.analysis.seg_table_utils import closest_row_to_value, filter_table
from flamingo_tools.export_data_utils import (
    crop_suffix, discover_crops, discover_source_types, find_crop_files, layer_kind, normalize_voxel_size,
    resolve_source_name, source_types, synapse_source_for_ihc,
)
from flamingo_tools.s3_utils import BUCKET_NAME, SERVICE_ENDPOINT, get_s3_path

DEFAULT_POSITIONS = {"length_fraction": {"apex": [0.15], "mid": [0.5], "base": [0.85]}}

# Colormaps cycled over the intensity channels of one crop, so that an overlay stays readable.
CHANNEL_COLORMAPS = ("green", "magenta", "cyan", "yellow")

# Named export groups. "reference" resolves the crop centers, "channels" are exported without
# filtering, "segmentations" with a component filter, and "synapses" adds a synapse export.
# All names may be aliases (see flamingo_tools.export_data_utils.SOURCE_ALIASES); "synapses" may
# also be "auto" to use the synapse source that is matched to the group's IHC segmentation.
# A group may also set "components", "roi_halo", "axis", "voxel_size" and "filter_cochlea" to
# override the run-level values.
EXPORT_GROUPS = {
    "sgn": {
        "reference": "SGN",
        "channels": ["PV"],
        "segmentations": ["SGN"],
        "synapses": None,
    },
    "ihc": {
        "reference": "IHC",
        "channels": ["VGlut3", "CTBP2"],
        "segmentations": ["IHC"],
        "synapses": "auto",
    },
}


def resolve_positions(ref_table: pd.DataFrame, positions: Dict[str, Dict[str, List[float]]]) -> List[dict]:
    """Resolve a position dict to concrete crop centers.

    Args:
        ref_table: Reference segmentation table. Must contain the column(s) named as
            top-level keys of `positions`, plus anchor_x/y/z and label_id.
        positions: Dict of {column: {label: [values, ...]}}, e.g.
            {"length_fraction": {"apex": [0.15], "mid": [0.5], "base": [0.85]}}. A label
            with multiple values produces one entry per value.

    Returns:
        List of dicts, one per (column, label, value) triple, each with keys "column",
        "label", "value", "label_id", "crop_center" ([x, y, z] in µm).
    """
    resolved = []
    for column, label_map in positions.items():
        for label, values in label_map.items():
            for value in values:
                row = closest_row_to_value(ref_table, column, value)
                resolved.append({
                    "column": column,
                    "label": label,
                    "value": value,
                    "label_id": int(row["label_id"]),
                    "crop_center": [float(row["anchor_x"]), float(row["anchor_y"]), float(row["anchor_z"])],
                })
    return resolved


def resolve_reference_positions(
    cochlea: str,
    reference_seg: str,
    positions: Dict[str, Dict[str, List[float]]],
    component_list: Optional[List[int]] = None,
) -> List[dict]:
    """Resolve crop centers from the table of a reference segmentation on S3.

    Args:
        cochlea: Cochlea name on S3 bucket.
        reference_seg: Segmentation source name, e.g. "SGN_v2". The table is read from
            <cochlea>/tables/<reference_seg>/default.tsv.
        positions: Dict of {column: {label: [values, ...]}}, see `resolve_positions`.
        component_list: Optional component_labels filter applied before matching positions.

    Returns:
        List of resolved positions, see `resolve_positions`.
    """
    internal_path = os.path.join(cochlea, "tables", reference_seg, "default.tsv")
    tsv_path, fs = get_s3_path(internal_path, bucket_name=BUCKET_NAME, service_endpoint=SERVICE_ENDPOINT)
    with fs.open(tsv_path, "r") as f:
        ref_table = pd.read_csv(f, sep="\t")

    if component_list is not None:
        ref_table = filter_table(ref_table, column_subset=component_list)

    return resolve_positions(ref_table, positions)


def _run_grid_export(fn, cochlea, scale, output_folder, crop_center, roi_halo, axis, suffix, voxel_size, extra):
    """Adapter for export_lower_resolution/_marker/_subtypes: identical (scale: List[int], ...) signature."""
    kwargs = dict(cochlea=cochlea, scale=scale, output_folder=output_folder, crop_center=crop_center,
                  roi_halo=roi_halo, axis=axis, suffix=suffix, voxel_size=voxel_size)
    kwargs.update(extra)
    fn(**kwargs)


def _run_synapse_export(fn, cochlea, scale, output_folder, crop_center, roi_halo, axis, suffix, voxel_size, extra):
    """Adapter for export_synapse_detections: takes `scales` instead of `scale`."""
    kwargs = dict(cochlea=cochlea, scales=scale, output_folder=output_folder, crop_center=crop_center,
                  roi_halo=roi_halo, axis=axis, suffix=suffix, voxel_size=voxel_size)
    kwargs.update(extra)
    fn(**kwargs)


def _run_frequency_export(fn, cochlea, scale, output_folder, crop_center, roi_halo, axis, suffix, voxel_size, extra):
    """Adapter for export_frequency_mapping: only accepts a single scale value, so loop over the list."""
    for s in scale:
        kwargs = dict(cochlea=cochlea, scale=s, output_folder=output_folder, crop_center=crop_center,
                      roi_halo=roi_halo, axis=axis, suffix=suffix, voxel_size=voxel_size)
        kwargs.update(extra)
        fn(**kwargs)


EXPORT_DISPATCH = {
    "lower_resolution": (_run_grid_export, export_lower_resolution.export_lower_resolution),
    "marker": (_run_grid_export, export_lower_resolution_marker.export_lower_resolution),
    "subtypes": (_run_grid_export, export_lower_resolution_subtypes.export_lower_resolution),
    "synapse": (_run_synapse_export, export_synapse_detections.export_synapse_detections),
    "frequency": (_run_frequency_export, export_frequency_mapping.export_frequency_mapping),
}

# The keyword argument that names the exported sources, per export function.
SOURCE_KWARGS = {
    "lower_resolution": "channels",
    "marker": "channels",
    "subtypes": "stains",
    "synapse": "synapse_name",
    "frequency": "source_name",
}


def describe_pass(key: str, kwargs: dict) -> str:
    """Describe the sources that one export pass writes.

    Args:
        key: Key into EXPORT_DISPATCH.
        kwargs: Keyword arguments of the pass.

    Returns:
        One line naming the export function and its sources.
    """
    names = kwargs.get(SOURCE_KWARGS[key])
    if names is None:
        parts = [f"'{key}' with its default sources"]
    else:
        parts = [f"'{key}' for {names if isinstance(names, list) else [names]}"]

    if kwargs.get("filter_by_components") is not None:
        parts.append(f"filtered by components {kwargs['filter_by_components']}")
    if kwargs.get("filter_cochlea_channels") is not None:
        parts.append(f"masked by {kwargs['filter_cochlea_channels']}")
    if key == "synapse" and kwargs.get("reference_ihcs") is not None:
        parts.append(f"matched to {kwargs['reference_ihcs']}")
    return ", ".join(parts)


def build_group_passes(sources: Dict[str, str], group: dict) -> List[Tuple[str, dict]]:
    """Expand an export group to the export passes it needs, with all source names resolved.

    The channels and the segmentations of a group need separate passes, because
    export_lower_resolution applies `filter_by_components` to every channel it is given.

    Args:
        sources: Source names and types of the cochlea, see flamingo_tools.export_data_utils.
        group: Export group definition, see EXPORT_GROUPS.

    Returns:
        List of (EXPORT_DISPATCH key, keyword arguments) tuples.
    """
    components = group.get("components") or [1]
    overrides = {key: group[key] for key in ("roi_halo", "axis", "voxel_size") if group.get(key) is not None}

    channels = [resolve_source_name(sources, name, "image") for name in group.get("channels") or []]
    segmentations = [resolve_source_name(sources, name, "segmentation") for name in group.get("segmentations") or []]

    passes = []
    if channels:
        kwargs = dict(channels=channels, **overrides)
        if group.get("filter_cochlea"):
            if not segmentations:
                raise ValueError("A group with 'filter_cochlea' requires at least one segmentation.")
            kwargs.update(filter_cochlea_channels=segmentations,
                          filter_sgn_components=components, filter_ihc_components=components)
        passes.append(("lower_resolution", kwargs))

    if segmentations:
        passes.append(("lower_resolution", dict(channels=segmentations, filter_by_components=components, **overrides)))

    synapses = group.get("synapses")
    if synapses:
        reference_ihcs = [seg for seg in segmentations if "IHC" in seg.upper()]
        if not reference_ihcs:
            raise ValueError("A group with a synapse export requires an IHC segmentation as reference.")
        synapse_name = (
            synapse_source_for_ihc(sources, reference_ihcs[0]) if synapses == "auto"
            else resolve_source_name(sources, synapses, "spots")
        )
        passes.append(("synapse", dict(
            synapse_name=synapse_name, reference_ihcs=reference_ihcs[0],
            filter_ihc_components=components, **overrides,
        )))

    if not passes:
        raise ValueError("The group does not contain any channel, segmentation or synapse export.")
    return passes


def describe_position(entry: dict) -> str:
    """Describe a crop position for a print or a napari window title.

    Args:
        entry: Resolved position, see `resolve_positions`, or a crop that was discovered on
            disk, see `flamingo_tools.export_data_utils.discover_crops`, which has no column.

    Returns:
        The position label, with the column and value it was resolved from where they are known.
    """
    label = entry["label"] or "crop"
    if entry.get("column") is None:
        return label
    return f"{label} ({entry['column']}={entry['value']})"


def view_crops(
    cochlea: str,
    folder: str,
    resolved_positions: List[dict],
    scale: List[int],
    sources: Dict[str, str],
    axis: Optional[int] = None,
    voxel_size: Sequence[float] = (0.38, 0.38, 0.38),
    label: Optional[str] = None,
):
    """Open exported crops in napari, one crop after another.

    Each viewer holds every layer of one crop at one scale level. The function blocks per crop, so
    the next crop opens once the current viewer is closed.

    Args:
        cochlea: Cochlea name, i.e. the sub-folder the export functions write to.
        folder: Output folder that was used for the export.
        resolved_positions: Output of `resolve_positions`.
        scale: Scale level(s) to open.
        sources: Source names and types of the cochlea, used to tell images from labels.
        axis: Axis that was used for the export, needed to find the files. A position may
            override it with its own "axis" entry.
        voxel_size: Voxel size of the data in micrometer as [x, y, z], used as the layer scale.
        label: Optional name of the export group, shown in the window title.
    """
    import napari

    voxel_size = normalize_voxel_size(voxel_size)
    title_prefix = cochlea if label is None else f"{cochlea} | {label}"

    for entry in resolved_positions:
        entry_axis = entry.get("axis", axis)
        suffix = crop_suffix(entry["crop_center"], entry_axis, entry["label"])
        crops = find_crop_files(os.path.join(folder, cochlea), entry["crop_center"], entry_axis, entry["label"])

        for s in scale:
            scale_folder = f"scale{s}"
            paths_per_folder = {
                crop_folder: paths for crop_folder, paths in crops.items()
                if os.path.basename(crop_folder) == scale_folder
                or os.path.basename(crop_folder).startswith(f"{scale_folder}_")
            }
            if not paths_per_folder:
                print(f"No exported file for '{describe_position(entry)}' at {scale_folder} in {folder}.")
                continue

            layer_scale_zyx = np.array(voxel_size[::-1]) * 2 ** s
            for crop_folder, paths in paths_per_folder.items():
                viewer = napari.Viewer()
                colormaps = cycle(CHANNEL_COLORMAPS)

                # The intensity channels come first, so that the labels are drawn on top of them.
                paths = sorted(paths, key=lambda path: (layer_kind(sources, os.path.basename(path)), path))
                for path in paths:
                    file_name = os.path.basename(path)
                    name = file_name[:-len(".tif")].replace(suffix, "")
                    data = tifffile.imread(path)
                    # Drop the single-pixel axis of a 2D slice, so that the crop is shown as an image
                    # instead of a one-pixel-wide volume.
                    layer_scale = (
                        tuple(sc for sc, dim in zip(layer_scale_zyx, data.shape) if dim != 1)
                        if data.ndim == 3 else None
                    )
                    data = np.squeeze(data)

                    if layer_kind(sources, file_name) == "labels":
                        viewer.add_labels(data.astype("uint32"), name=name, scale=layer_scale)
                    else:
                        viewer.add_image(data, name=name, scale=layer_scale, blending="additive",
                                         colormap=next(colormaps))

                viewer.title = f"{title_prefix} | {describe_position(entry)} | {os.path.basename(crop_folder)}"
                napari.run()


def group_source_kinds(group: dict) -> Dict[str, str]:
    """Map the source names of an export group to their MoBIE source types.

    The names are not resolved against the cochlea, so an alias stays an alias. This types the
    exported files of a group without reading the dataset on S3, see
    `flamingo_tools.export_data_utils.discover_source_types`.

    Args:
        group: Export group definition, see EXPORT_GROUPS.

    Returns:
        Dict of {source name or alias: source type}.
    """
    kinds = {name: "image" for name in group.get("channels") or []}
    kinds.update({name: "segmentation" for name in group.get("segmentations") or []})

    synapses = group.get("synapses")
    if synapses is not None:
        kinds["synapses" if synapses == "auto" else synapses] = "spots"
    return kinds


def view_exported_crops(
    cochlea: str,
    folder: str,
    scale: List[int],
    declared_sources: Optional[Dict[str, str]] = None,
    axis: Optional[int] = None,
    voxel_size: Sequence[float] = (0.38, 0.38, 0.38),
    label: Optional[str] = None,
):
    """Open the crops that are already exported in a folder, without reading anything from S3.

    The crop centers, the position labels, the crop axis and the source types all come from the
    exported file names, so a finished export can be reviewed offline.

    Args:
        cochlea: Cochlea name, i.e. the sub-folder the export functions wrote to.
        folder: Output folder that was used for the export.
        scale: Scale level(s) to open.
        declared_sources: Optional {source name or alias: source type} of the export, used to
            tell images from labels, see `flamingo_tools.export_data_utils.discover_source_types`.
        axis: Axis that was used for the export. Only used for crops whose file name holds no axis.
        voxel_size: Voxel size of the data in micrometer as [x, y, z], used as the layer scale.
        label: Optional name of the export group, shown in the window title.
    """
    crop_folder = os.path.join(folder, cochlea)
    crops = discover_crops(crop_folder)
    if not crops:
        print(f"No exported crop found in {crop_folder}.")
        return

    sources = discover_source_types(crop_folder, declared=declared_sources)
    for entry in crops:
        print(f"  Viewing '{describe_position(entry)}' at crop_center (x, y, z) [µm] = {entry['crop_center']}")

    view_crops(cochlea, folder, crops, scale, sources, axis=axis, voxel_size=voxel_size, label=label)


def run_exports(
    cochlea: str,
    resolved_positions: List[dict],
    export_functions: List[str],
    scale: List[int],
    output_folder: str,
    roi_halo: Optional[List[int]] = None,
    axis: Optional[int] = None,
    json_info: Optional[Union[Dict[str, dict], List[Dict[str, dict]]]] = None,
    voxel_size: Sequence[float] = (0.38, 0.38, 0.38),
):
    """Run the requested export functions for every resolved position.

    Args:
        cochlea: Cochlea name on S3 bucket.
        resolved_positions: Output of `resolve_positions`.
        export_functions: Keys into EXPORT_DISPATCH to run at every position, e.g.
            ["lower_resolution", "synapse"]. Ignored when `json_info` is a list (see below).
        scale: Scale level(s) to export, applied to every selected export function.
        output_folder: Output folder, forwarded to every export function.
        roi_halo: Optional halo in pixels [x, y, z]. Required unless `axis` is given, or an
            export function's own entry below overrides it.
        axis: Optional axis (0=x, 1=y, 2=z) to crop as a single-pixel 2D slice at each
            position's center. Same override note as `roi_halo`.
        voxel_size: Voxel size of the data in micrometer as [x, y, z]. Same override note as
            `roi_halo`.
        json_info: Extra keyword arguments per export function, either:
            - a dict {export_function_key: {kwargs}}, applied to `export_functions` (looked
              up via `.get(key, {})`) -- e.g. {"synapse": {"synapse_name": "..."}}; or
            - a list of such dicts, one independent export "pass" per entry. In this mode
              `export_functions` is ignored: each entry's own key(s) determine which export
              function(s) run for that pass, so several distinct per-function configs (e.g.
              a combined SGN+IHC lower_resolution export plus separate IHC-only/SGN-only
              ones) can run from a single --json_info file instead of one invocation each.
            In both modes, a per-function kwargs dict may itself include "roi_halo"/"axis"/
            "voxel_size" keys to override the shared arguments above for that export function only.
    """
    if isinstance(json_info, list):
        passes = [(list(entry.keys()), entry) for entry in json_info]
    else:
        passes = [(export_functions, json_info or {})]

    for pass_functions, pass_json_info in passes:
        for key in pass_functions:
            if key not in EXPORT_DISPATCH:
                raise ValueError(f"Unknown export function '{key}'. Valid options: {list(EXPORT_DISPATCH)}")
            print("Exporting", describe_pass(key, pass_json_info.get(key, {})))

    for entry in resolved_positions:
        print(
            f"Position '{entry['label']}' ({entry['column']}={entry['value']}): "
            f"label_id={entry['label_id']}, crop_center (x, y, z) [µm] = {entry['crop_center']}"
        )

    for pass_functions, pass_json_info in passes:
        for entry in resolved_positions:
            for key in pass_functions:
                adapter, fn = EXPORT_DISPATCH[key]
                extra = pass_json_info.get(key, {})
                adapter(
                    fn, cochlea, scale, output_folder, entry["crop_center"], roi_halo, axis, entry["label"],
                    voxel_size, extra,
                )


def run_groups(
    cochlea: str,
    groups: Dict[str, dict],
    scale: List[int],
    output_folder: str,
    positions: Optional[Dict[str, Dict[str, List[float]]]] = None,
    components: Optional[List[int]] = None,
    roi_halo: Optional[List[int]] = None,
    axis: Optional[int] = None,
    voxel_size: Sequence[float] = (0.38, 0.38, 0.38),
    dry_run: bool = False,
    view: bool = False,
    view_only: bool = False,
):
    """Export the channels, segmentations and synapses of export groups at tonotopic positions.

    Args:
        cochlea: Cochlea name on S3 bucket.
        groups: Export groups to run, as {group name: group definition}, see EXPORT_GROUPS.
            Each group writes to its own sub-folder of `output_folder`.
        scale: Scale level(s) to export.
        output_folder: Root output folder.
        positions: Dict of {column: {label: [values, ...]}}. Default: DEFAULT_POSITIONS.
        components: Component filter for the reference table and the segmentation exports.
            Overrides the "components" entry of a group. Default: the group's value, or [1].
        roi_halo: Optional halo in pixels [x, y, z]. Required unless `axis` is given, or a
            group overrides it.
        axis: Optional axis (0=x, 1=y, 2=z) to crop as a single-pixel 2D slice at each
            position's center. Same override note as `roi_halo`.
        voxel_size: Voxel size of the data in micrometer as [x, y, z]. Same override note as
            `roi_halo`.
        dry_run: Only print the resolved sources, crop centers and export passes.
        view: Whether to open the crops of a group in napari once the group is exported.
        view_only: Whether to only open the crops that are already in the output folder, without
            exporting. Nothing is read from S3 in this mode, see `view_exported_crops`.
    """
    if positions is None:
        positions = DEFAULT_POSITIONS

    sources = None if view_only else source_types(cochlea)
    position_cache = {}

    for name, group in groups.items():
        group_folder = os.path.join(output_folder, name)
        group_axis = axis if group.get("axis") is None else group["axis"]
        group_voxel_size = voxel_size if group.get("voxel_size") is None else group["voxel_size"]

        if view_only:
            print(f"\nGroup '{name}'")
            view_exported_crops(cochlea, group_folder, scale, declared_sources=group_source_kinds(group),
                                axis=group_axis, voxel_size=group_voxel_size, label=name)
            continue

        group_components = components if components is not None else group.get("components") or [1]
        reference = resolve_source_name(sources, group["reference"], "segmentation")

        cache_key = (reference, tuple(group_components))
        if cache_key not in position_cache:
            position_cache[cache_key] = resolve_reference_positions(
                cochlea, reference, positions, group_components
            )
        resolved = position_cache[cache_key]
        passes = build_group_passes(sources, {**group, "components": group_components})

        if roi_halo is None and group_axis is None and group.get("roi_halo") is None:
            raise ValueError(f"Group '{name}' requires roi_halo or axis, either shared or in the group.")

        print(f"\nGroup '{name}': reference {reference}, components {group_components}")
        if dry_run:
            for key, kwargs in passes:
                print(f"  {key}: {kwargs}")
            for entry in resolved:
                print(
                    f"  Position '{entry['label']}' ({entry['column']}={entry['value']}): "
                    f"label_id={entry['label_id']}, crop_center (x, y, z) [µm] = {entry['crop_center']}"
                )
            continue

        run_exports(
            cochlea, resolved, [], scale, group_folder,
            roi_halo=roi_halo, axis=axis, json_info=[{key: kwargs} for key, kwargs in passes],
            voxel_size=voxel_size,
        )

        if view:
            view_crops(cochlea, group_folder, resolved, scale, sources,
                       axis=group_axis, voxel_size=group_voxel_size, label=name)


def export_by_position(
    cochlea: str,
    reference_seg: Optional[str],
    scale: List[int],
    output_folder: str,
    export_functions: List[str] = ["lower_resolution"],
    positions: Optional[Dict[str, Dict[str, List[float]]]] = None,
    component_list: Optional[List[int]] = None,
    roi_halo: Optional[List[int]] = None,
    axis: Optional[int] = None,
    json_info: Optional[Union[Dict[str, dict], List[Dict[str, dict]]]] = None,
    voxel_size: Sequence[float] = (0.38, 0.38, 0.38),
    view: bool = False,
    view_only: bool = False,
):
    """Export crops at tonotopic positions for a single cochlea.

    Args:
        cochlea: Cochlea name on S3 bucket.
        reference_seg: Segmentation source name used to resolve crop centers, e.g. "SGN_v2",
            or an alias such as "SGN" or "IHC". Not used with `view_only`.
        scale: Scale level(s) to export, applied to every selected export function.
        output_folder: Output folder, forwarded to every export function.
        export_functions: Keys into EXPORT_DISPATCH to run at every position. Default:
            ["lower_resolution"]. Ignored when `json_info` is a list -- see `run_exports`.
        positions: Dict of {column: {label: [values, ...]}}. Default: DEFAULT_POSITIONS
            ({"length_fraction": {"apex": [0.15], "mid": [0.5], "base": [0.85]}}).
        component_list: Optional component_labels filter applied to the reference table
            before matching positions.
        roi_halo: Optional halo in pixels [x, y, z]. Required unless `axis` is given, or an
            export function's own entry overrides it.
        axis: Optional axis (0=x, 1=y, 2=z) to crop as a single-pixel 2D slice at each
            position's center. Same override note as `roi_halo`.
        json_info: Extra keyword arguments per export function -- a dict or a list of dicts,
            one independent export pass per entry. See `run_exports` for the full semantics.
        voxel_size: Voxel size of the data in micrometer as [x, y, z]. Same override note as
            `roi_halo`.
        view: Whether to open the exported crops in napari.
        view_only: Whether to only open the crops that are already in the output folder, without
            exporting. Nothing is read from S3 in this mode, see `view_exported_crops`. The
            reference segmentation, the positions and the components are then not used.
    """
    if positions is None:
        positions = DEFAULT_POSITIONS

    if view_only:
        view_exported_crops(cochlea, output_folder, scale, axis=axis, voxel_size=voxel_size)
        return

    sources = source_types(cochlea)
    reference_seg = resolve_source_name(sources, reference_seg, "segmentation")
    resolved = resolve_reference_positions(cochlea, reference_seg, positions, component_list)

    run_exports(
        cochlea, resolved, export_functions, scale, output_folder,
        roi_halo=roi_halo, axis=axis, json_info=json_info, voxel_size=voxel_size,
    )

    if view:
        view_crops(cochlea, output_folder, resolved, scale, sources, axis=axis, voxel_size=voxel_size)


def select_groups(args, parser) -> Dict[str, dict]:
    """Build the export groups for a CLI invocation, applying the definition file and overrides."""
    definitions = dict(EXPORT_GROUPS)
    if args.groups_json is not None:
        with open(args.groups_json) as f:
            definitions.update(json.load(f))

    unknown = [name for name in args.groups if name not in definitions]
    if unknown:
        parser.error(f"Unknown export group(s) {unknown}. Available: {sorted(definitions)}")

    overrides = {}
    if args.channels is not None:
        overrides["channels"] = args.channels
    if args.segmentations is not None:
        overrides["segmentations"] = args.segmentations
    if args.synapses is not None:
        overrides["synapses"] = args.synapses
    if args.no_synapses:
        overrides["synapses"] = None
    if args.reference_seg is not None:
        overrides["reference"] = args.reference_seg

    group_specific = {"channels", "segmentations", "synapses"}
    if len(args.groups) > 1 and group_specific.intersection(overrides):
        parser.error(
            "--channels/--segmentations/--synapses are specific to one group. "
            "Select a single group, or define the groups with --groups_json."
        )

    return {name: {**definitions[name], **overrides} for name in args.groups}


def main():
    parser = argparse.ArgumentParser(
        description="Export 2D crops at tonotopic positions (default: apex/mid/base) for a "
        "single cochlea, either from named export groups (--groups) or by dispatching export "
        "functions directly (--export_functions/--json_info).")
    parser.add_argument("--cochlea", "-c", required=True)
    parser.add_argument("--scale", "-s", nargs="+", type=int, required=True)
    parser.add_argument("--output_folder", "-o", required=True)
    parser.add_argument("--groups", nargs="+", default=None,
                        help=f"Export group(s) to run. Available: {sorted(EXPORT_GROUPS)}. Every group "
                        "resolves its own crop centers and writes to its own sub-folder of the output "
                        "folder.")
    parser.add_argument("--groups_json", type=str, default=None,
                        help="JSON file with export group definitions, merged over the built-in ones. "
                        "Same schema as EXPORT_GROUPS.")
    parser.add_argument("--reference_seg", default=None,
                        help="Segmentation source name used to resolve crop centers, e.g. SGN_v2, or an "
                        "alias such as SGN or IHC. Required with --export_functions. With --groups it "
                        "overrides the reference of the selected group(s).")
    parser.add_argument("--channels", nargs="+", type=str, default=None,
                        help="Override the channels of the selected group. Requires a single --groups entry.")
    parser.add_argument("--segmentations", nargs="+", type=str, default=None,
                        help="Override the segmentations of the selected group. Requires a single --groups entry.")
    parser.add_argument("--synapses", type=str, default=None,
                        help="Override the synapse source of the selected group, or 'auto' for the source that "
                        "is matched to the group's IHC segmentation. Requires a single --groups entry.")
    parser.add_argument("--no_synapses", action="store_true",
                        help="Do not export synapses for the selected group(s).")
    parser.add_argument("--export_functions", nargs="+", default=None,
                        choices=list(EXPORT_DISPATCH),
                        help="Export function(s) to run at each position, without group expansion. "
                        "Default: lower_resolution")
    parser.add_argument("--positions_json", type=str, default=None,
                        help="Optional JSON file overriding the default position dict "
                        f"{DEFAULT_POSITIONS}.")
    parser.add_argument("-C", "--components", nargs="+", type=int, default=None,
                        help="Component filter for the reference table and the segmentation exports. "
                        "Overrides the 'components' entry of a group. Default: 1")
    parser.add_argument("--roi_halo", nargs=3, type=int, default=None,
                        help="Halo around each position's crop center as halo_x halo_y halo_z in pixels "
                        "at the target scale. Optional when --axis is given (the whole plane is cropped "
                        "in that case), or when every relevant group or --json_info entry supplies its "
                        "own roi_halo/axis.")
    parser.add_argument("--axis", type=int, choices=[0, 1, 2], default=None,
                        help="Axis (0=x, 1=y, 2=z) to crop as a single-pixel 2D slice at each "
                        "position's center.")
    parser.add_argument("-j", "--json_info", type=str, default=None,
                        help="JSON file with extra keyword arguments per export function, keyed by "
                        "export-function name, e.g. {\"synapse\": {\"synapse_name\": \"...\"}}. Can "
                        "also be a JSON list of such dicts, one independent export pass per entry -- "
                        "each entry's own key(s) then determine which export function(s) run for that "
                        "pass, and --export_functions is ignored.")
    parser.add_argument("-v", "--voxel_size", type=float, nargs="+", default=[0.38, 0.38, 0.38],
                        help="Voxel size of input in micrometer. Default: 0.38 0.38 0.38")
    parser.add_argument("--dry_run", action="store_true",
                        help="Only print the resolved source names, crop centers and export passes.")
    parser.add_argument("--view", action="store_true",
                        help="Open the exported crops in napari, one crop after another. The crops of a group "
                        "open once the group is exported.")
    parser.add_argument("--view_only", action="store_true",
                        help="Only open the crops that are already in the output folder, without exporting. "
                        "The crop centers, position labels and source types are taken from the exported file "
                        "names, so this mode does not read anything from S3.")
    args = parser.parse_args()

    positions = DEFAULT_POSITIONS
    if args.positions_json is not None:
        with open(args.positions_json) as f:
            positions = json.load(f)

    if args.dry_run and (args.view or args.view_only):
        parser.error("--dry_run cannot be combined with --view or --view_only.")

    if args.groups is not None:
        if args.export_functions is not None or args.json_info is not None:
            parser.error("--groups cannot be combined with --export_functions or --json_info.")
        run_groups(
            cochlea=args.cochlea,
            groups=select_groups(args, parser),
            scale=args.scale,
            output_folder=args.output_folder,
            positions=positions,
            components=args.components,
            roi_halo=args.roi_halo,
            axis=args.axis,
            voxel_size=args.voxel_size,
            dry_run=args.dry_run,
            view=args.view,
            view_only=args.view_only,
        )
        return

    if args.reference_seg is None and not args.view_only:
        parser.error("--reference_seg is required without --groups.")
    for name in ("channels", "segmentations", "synapses"):
        if getattr(args, name) is not None:
            parser.error(f"--{name} requires --groups.")
    if args.dry_run:
        parser.error("--dry_run requires --groups.")

    json_info = None
    if args.json_info is not None:
        with open(args.json_info) as f:
            json_info = json.load(f)

    export_by_position(
        cochlea=args.cochlea,
        reference_seg=args.reference_seg,
        scale=args.scale,
        output_folder=args.output_folder,
        export_functions=args.export_functions or ["lower_resolution"],
        positions=positions,
        component_list=args.components or [1],
        roi_halo=args.roi_halo,
        axis=args.axis,
        json_info=json_info,
        voxel_size=args.voxel_size,
        view=args.view,
        view_only=args.view_only,
    )


if __name__ == "__main__":
    main()
