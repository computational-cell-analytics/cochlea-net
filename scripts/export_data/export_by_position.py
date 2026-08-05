"""Export 2D crops at standard tonotopic positions (default: apex/mid/base) for a single
cochlea, dispatching to one or more of the export_data scripts in this folder.

Crop centers are resolved once from a reference segmentation table (the same lookup that
powers flamingo_tools.table_info) and reused unchanged across every selected
--export_functions, since crop centers are physical µm coordinates shared across all
co-registered channels of a cochlea.
"""
import argparse
import json
import os
from typing import Dict, List, Optional, Union

import pandas as pd

import export_frequency_mapping
import export_lower_resolution
import export_lower_resolution_marker
import export_lower_resolution_subtypes
import export_synapse_detections

from flamingo_tools.analysis.seg_table_utils import closest_row_to_value, filter_table
from flamingo_tools.s3_utils import BUCKET_NAME, SERVICE_ENDPOINT, get_s3_path

DEFAULT_POSITIONS = {"length_fraction": {"apex": [0.15], "mid": [0.5], "base": [0.85]}}


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


def _run_grid_export(fn, cochlea, scale, output_folder, crop_center, roi_halo, axis, suffix, extra):
    """Adapter for export_lower_resolution/_marker/_subtypes: identical (scale: List[int], ...) signature."""
    kwargs = dict(cochlea=cochlea, scale=scale, output_folder=output_folder,
                  crop_center=crop_center, roi_halo=roi_halo, axis=axis, suffix=suffix)
    kwargs.update(extra)
    fn(**kwargs)


def _run_synapse_export(fn, cochlea, scale, output_folder, crop_center, roi_halo, axis, suffix, extra):
    """Adapter for export_synapse_detections: takes `scales` instead of `scale`."""
    kwargs = dict(cochlea=cochlea, scales=scale, output_folder=output_folder,
                  crop_center=crop_center, roi_halo=roi_halo, axis=axis, suffix=suffix)
    kwargs.update(extra)
    fn(**kwargs)


def _run_frequency_export(fn, cochlea, scale, output_folder, crop_center, roi_halo, axis, suffix, extra):
    """Adapter for export_frequency_mapping: only accepts a single scale value, so loop over the list."""
    for s in scale:
        kwargs = dict(cochlea=cochlea, scale=s, output_folder=output_folder,
                      crop_center=crop_center, roi_halo=roi_halo, axis=axis, suffix=suffix)
        kwargs.update(extra)
        fn(**kwargs)


EXPORT_DISPATCH = {
    "lower_resolution": (_run_grid_export, export_lower_resolution.export_lower_resolution),
    "marker": (_run_grid_export, export_lower_resolution_marker.export_lower_resolution),
    "subtypes": (_run_grid_export, export_lower_resolution_subtypes.export_lower_resolution),
    "synapse": (_run_synapse_export, export_synapse_detections.export_synapse_detections),
    "frequency": (_run_frequency_export, export_frequency_mapping.export_frequency_mapping),
}


def run_exports(
    cochlea: str,
    resolved_positions: List[dict],
    export_functions: List[str],
    scale: List[int],
    output_folder: str,
    roi_halo: Optional[List[int]] = None,
    axis: Optional[int] = None,
    json_info: Optional[Union[Dict[str, dict], List[Dict[str, dict]]]] = None,
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
        json_info: Extra keyword arguments per export function, either:
            - a dict {export_function_key: {kwargs}}, applied to `export_functions` (looked
              up via `.get(key, {})`) -- e.g. {"synapse": {"synapse_name": "..."}}; or
            - a list of such dicts, one independent export "pass" per entry. In this mode
              `export_functions` is ignored: each entry's own key(s) determine which export
              function(s) run for that pass, so several distinct per-function configs (e.g.
              a combined SGN+IHC lower_resolution export plus separate IHC-only/SGN-only
              ones) can run from a single --json_info file instead of one invocation each.
            In both modes, a per-function kwargs dict may itself include "roi_halo"/"axis"
            keys to override the shared arguments above for that export function only.
    """
    if isinstance(json_info, list):
        passes = [(list(entry.keys()), entry) for entry in json_info]
    else:
        passes = [(export_functions, json_info or {})]

    for pass_functions, pass_json_info in passes:
        for entry in resolved_positions:
            print(
                f"Position '{entry['label']}' ({entry['column']}={entry['value']}): "
                f"label_id={entry['label_id']}, crop_center (x, y, z) [µm] = {entry['crop_center']}"
            )
            for key in pass_functions:
                if key not in EXPORT_DISPATCH:
                    raise ValueError(f"Unknown export function '{key}'. Valid options: {list(EXPORT_DISPATCH)}")
                adapter, fn = EXPORT_DISPATCH[key]
                extra = pass_json_info.get(key, {})
                adapter(
                    fn, cochlea, scale, output_folder, entry["crop_center"], roi_halo, axis, entry["label"], extra
                )


def export_by_position(
    cochlea: str,
    reference_seg: str,
    scale: List[int],
    output_folder: str,
    export_functions: List[str] = ["lower_resolution"],
    positions: Optional[Dict[str, Dict[str, List[float]]]] = None,
    component_list: Optional[List[int]] = None,
    roi_halo: Optional[List[int]] = None,
    axis: Optional[int] = None,
    json_info: Optional[Union[Dict[str, dict], List[Dict[str, dict]]]] = None,
):
    """Export crops at tonotopic positions for a single cochlea.

    Args:
        cochlea: Cochlea name on S3 bucket.
        reference_seg: Segmentation channel name used to resolve crop centers, e.g. "SGN_v2".
            The table is read from <cochlea>/tables/<reference_seg>/default.tsv on S3.
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
    """
    if positions is None:
        positions = DEFAULT_POSITIONS

    internal_path = os.path.join(cochlea, "tables", reference_seg, "default.tsv")
    tsv_path, fs = get_s3_path(internal_path, bucket_name=BUCKET_NAME, service_endpoint=SERVICE_ENDPOINT)
    with fs.open(tsv_path, "r") as f:
        ref_table = pd.read_csv(f, sep="\t")

    if component_list is not None:
        ref_table = filter_table(ref_table, column_subset=component_list)

    resolved = resolve_positions(ref_table, positions)
    run_exports(
        cochlea, resolved, export_functions, scale, output_folder,
        roi_halo=roi_halo, axis=axis, json_info=json_info,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Export 2D crops at tonotopic positions (default: apex/mid/base) for a "
        "single cochlea, dispatching to one or more export_data scripts.")
    parser.add_argument("--cochlea", "-c", required=True)
    parser.add_argument("--reference_seg", required=True,
                        help="Segmentation channel name used to resolve crop centers, e.g. SGN_v2.")
    parser.add_argument("--scale", "-s", nargs="+", type=int, required=True)
    parser.add_argument("--output_folder", "-o", required=True)
    parser.add_argument("--export_functions", nargs="+", default=["lower_resolution"],
                        choices=list(EXPORT_DISPATCH),
                        help="Export function(s) to run at each position. Default: lower_resolution")
    parser.add_argument("--positions_json", type=str, default=None,
                        help="Optional JSON file overriding the default position dict "
                        f"{DEFAULT_POSITIONS}.")
    parser.add_argument("-C", "--components", nargs="+", type=int, default=[1],
                        help="Component filter applied to the reference table before matching positions.")
    parser.add_argument("--roi_halo", nargs=3, type=int, default=None,
                        help="Halo around each position's crop center as halo_x halo_y halo_z in pixels "
                        "at the target scale. Optional when --axis is given (the whole plane is cropped "
                        "in that case), or when every relevant --json_info entry supplies its own "
                        "roi_halo/axis.")
    parser.add_argument("--axis", type=int, choices=[0, 1, 2], default=None,
                        help="Axis (0=x, 1=y, 2=z) to crop as a single-pixel 2D slice at each "
                        "position's center.")
    parser.add_argument("-j", "--json_info", type=str, default=None,
                        help="JSON file with extra keyword arguments per export function, keyed by "
                        "export-function name, e.g. {\"synapse\": {\"synapse_name\": \"...\"}}. Can "
                        "also be a JSON list of such dicts, one independent export pass per entry -- "
                        "each entry's own key(s) then determine which export function(s) run for that "
                        "pass, and --export_functions is ignored.")
    args = parser.parse_args()

    positions = DEFAULT_POSITIONS
    if args.positions_json is not None:
        with open(args.positions_json) as f:
            positions = json.load(f)

    json_info = None
    if args.json_info is not None:
        with open(args.json_info) as f:
            json_info = json.load(f)

    export_by_position(
        cochlea=args.cochlea,
        reference_seg=args.reference_seg,
        scale=args.scale,
        output_folder=args.output_folder,
        export_functions=args.export_functions,
        positions=positions,
        component_list=args.components,
        roi_halo=args.roi_halo,
        axis=args.axis,
        json_info=json_info,
    )


if __name__ == "__main__":
    main()
