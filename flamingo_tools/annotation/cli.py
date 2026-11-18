"""The command line interface of the annotation correction: `flamingo_tools.correct_annotation`."""
import argparse
import os
from dataclasses import dataclass, fields
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import tifffile

from .controller import Controller, Settings
from .crop_reference import (
    CropInfo, cell_type_of, crop_candidates, ihc_sources, load_reference, read_reference, reference_cache_path,
)
from .ids import compact
from .issues import Item, cell_voxels, collect_items, detect, missing_proposal, read_issue_table, table_findings
from .region import apply_region, propose_region, region_items, row_context, source_variants
from .session import Session, array_sha1


@dataclass
class Correction:
    """A prepared correction: the loop and what the user interface needs.

    The base is the segmentation that the loop corrects. The comparison is the other one (S3 or the annotation),
    with its file ids.
    """
    controller: Controller
    crop_name: str
    original: np.ndarray
    output_path: str
    table_path: str
    comparison: Optional[np.ndarray]
    comparison_name: Optional[str]


def species_of(cochlea: Optional[str]) -> str:
    return "gerbil" if cochlea and cochlea.startswith("G") else "mouse"


def reference_list(reference: str) -> Optional[Sequence[str]]:
    """The S3 sources of the --reference option: None for 'auto' (the default order), else the listed sources."""
    if reference == "auto":
        return None
    return [name.strip() for name in reference.split(",") if name.strip()]


def load_region_sources(
    info: Optional[CropInfo], names: Union[str, Sequence[str]], cache_dir: str,
) -> Dict[str, np.ndarray]:
    """The S3 IHC model segmentations of a crop for the region proposal: 'all', or the listed sources."""
    from flamingo_tools.export_data_utils import source_types

    if info is None:
        print("The crop has no S3 position, so the region proposal uses the annotation only.")
        return {}
    available = ihc_sources(source_types(info.cochlea))
    if names != "all":
        missing = [name for name in names if name not in available]
        if missing:
            print(f"{info.cochlea} has no segmentation {', '.join(missing)}. It has: {', '.join(available)}.")
        available = [name for name in available if name in names]
    print(f"Region sources: {', '.join(available) or 'none'}.")
    return {name: read_reference(info, name, cache_dir) for name in available}


def _region_items(
    labels: np.ndarray,
    session: Session,
    regions: Sequence[Tuple[int, int]],
    id_labels: np.ndarray,
    id_original: np.ndarray,
    sources: Dict[str, np.ndarray],
    base_source: str,
    voxel_size: float,
) -> Tuple[List[Item], object]:
    """Propose the cells of each region, apply them to the labels as one commit each, and return their items."""
    def compact_id(file_id: int) -> int:
        index = int(np.searchsorted(id_original, file_id))
        if index >= len(id_original) or int(id_original[index]) != file_id:
            raise ValueError(f"The region label {file_id} is not in the labels.")
        return index

    print("Row correction of the region sources ...")
    path, row = row_context(id_labels, sources, voxel_size)
    variants = source_variants(sources, path, row.reference, voxel_size)
    next_label = [int(labels.max()) + 1]

    def new_label() -> int:
        next_label[0] += 1
        return next_label[0] - 1

    items = []
    for file_a, file_b in regions:
        ends = (compact_id(file_a), compact_id(file_b))
        proposal = propose_region(labels, id_labels, ends, variants, voxel_size, base_source, path_row=(path, row),
                                  correct=None)
        box, after, cells = apply_region(labels, proposal, new_label, voxel_size)
        start = tuple(sl.start for sl in box)
        session.commit(start, labels[box], after, {"item": -1, "action": "region", "label_id": 0,
                                                   "reasons": f"region {file_a}-{file_b}"})
        labels[box] = after
        print(f"Region {file_a}-{file_b}: {len(cells)} cells.")
        items += region_items(labels, cells, proposal.end_labels, first_id=len(items))
    return items, row


def prepare_correction(
    image_path: str,
    output_folder: str,
    annotation_path: Optional[str] = None,
    edit_annotation: bool = False,
    cell_type: Optional[str] = None,
    reference: str = "auto",
    issues: Optional[str] = None,
    cochlea: Optional[str] = None,
    center: Optional[Sequence[float]] = None,
    restart: bool = False,
    settings: Optional[Settings] = None,
    regions: Optional[Sequence[Tuple[int, int]]] = None,
    region_sources: Union[str, Sequence[str]] = "all",
) -> Correction:
    """Load a crop, find the items to correct, and start or resume its session.

    The S3 segmentation is the base, and the annotation (if given) is the comparison. With `edit_annotation`, the
    annotation is the base and the S3 segmentation is the comparison.

    With `regions`, the tool proposes new cells for each row stretch between two cells (`region.propose_region`),
    and the items are these cells and the end cells. The detectors do not run, but the issue table adds its rows.

    Args:
        image_path: The image tif of the crop. Its file name without '.tif' is the crop name.
        output_folder: The folder for the corrected segmentation, the change table and the session.
        annotation_path: The annotation tif.
        edit_annotation: Correct the annotation instead of the S3 segmentation.
        cell_type: 'ihc' or 'sgn'. The default comes from the stain in the crop name.
        reference: 'auto' for the default S3 sources (`crop_reference.REFERENCE_ORDER`), 'none', or a
            comma-separated list of S3 segmentation sources, tried in order.
        issues: An issue table, see `issues.read_issue_table`. Its label ids are annotation ids.
        cochlea: The cochlea of the crop, if the crop name does not give it.
        center: The crop center in micrometer in (x, y, z) order, if the crop name does not give it.
        restart: Discard an existing session of the crop.
        settings: The settings of the loop.
        regions: Pairs of cells (ids of the annotation, or of the base without an annotation) that bound a row
            stretch to segment again. IHC only.
        region_sources: 'all' IHC model segmentations of the cochlea on S3, or a list of sources.

    Returns:
        The prepared correction.
    """
    settings = settings or Settings()
    crop_name = os.path.basename(image_path)[:-len(".tif")] if image_path.endswith(".tif") else \
        os.path.basename(image_path)
    image = tifffile.imread(image_path)
    found_cochlea, _, stain, _ = crop_candidates(crop_name)
    cell_type = cell_type or cell_type_of(stain)
    if cell_type is None:
        raise ValueError(f"The cell type of {crop_name} is unknown. Give it with --cell_type.")
    if edit_annotation and annotation_path is None:
        raise ValueError("--edit_annotation needs the annotation (-a).")
    annotation = tifffile.imread(annotation_path) if annotation_path else None
    if annotation is not None and annotation.shape != image.shape:
        raise ValueError(f"The annotation shape {annotation.shape} differs from the image shape {image.shape}.")
    os.makedirs(output_folder, exist_ok=True)
    session_folder = os.path.join(output_folder, f"{crop_name}_session")
    resume = Session.exists(session_folder) and not restart

    if regions and cell_type != "ihc":
        raise ValueError("The region proposal works for IHC crops only.")
    s3, s3_name, info = None, None, None
    if reference != "none":
        s3, s3_name, info = load_reference(
            crop_name, image, cell_type, cache_dir=session_folder if resume else None,
            candidates=reference_list(reference), cochlea=cochlea, center=center, voxel_size=settings.voxel_size,
        )
    if edit_annotation:
        base, comparison, comparison_name, tag = annotation, s3, s3_name, "s3"
    else:
        if s3 is None:
            raise ValueError(f"{crop_name} has no S3 segmentation to correct. Use --edit_annotation with -a to "
                             "correct the annotation instead.")
        base, comparison, comparison_name, tag = s3, annotation, "annotation", "annotation"

    base_sha1 = array_sha1(base)
    if resume:
        session = Session.open(session_folder, base_sha1)
    else:
        session = Session.create(session_folder, base_sha1, {
            "crop": crop_name, "cell_type": cell_type, "issues": issues,
            "base": "annotation" if edit_annotation else f"s3:{s3_name}",
        })
        if s3 is not None:
            tifffile.imwrite(reference_cache_path(session_folder, crop_name, s3_name), s3, compression="zlib")

    labels, original = compact(base)
    comparison_compact = compact(comparison)[0] if comparison is not None else None
    species = species_of(cochlea or found_cochlea)
    issue_rows = read_issue_table(issues, crop_name) if issues else None
    # The issue table holds annotation ids: they are the base ids, or they map to the S3 base by overlap.
    table_ids = {"original": original} if edit_annotation else {"source_labels": annotation}
    items, row = [], None
    if regions and not resume:
        if annotation is None or edit_annotation:
            id_labels, id_original = labels, original
        else:
            id_labels, id_original = compact(annotation)
        sources = load_region_sources(info, region_sources, session_folder)
        if s3_name in sources and not edit_annotation:
            sources[s3_name] = labels
        else:
            sources = {name: compact(source)[0] for name, source in sources.items()}
        base_source = "annotation" if edit_annotation else s3_name
        items, row = _region_items(labels, session, regions, id_labels, id_original, sources, base_source,
                                   settings.voxel_size)
        if issue_rows is not None and len(issue_rows):
            findings = table_findings(issue_rows, labels, cell_voxels(labels), table_ids.get("original"),
                                      table_ids.get("source_labels"))
            known = {item.label_id for item in items}
            extra = [item for item in collect_items(findings, None, labels.shape) if item.label_id not in known]
            for offset, item in enumerate(extra):
                item.item_id, item.order = len(items) + offset, float(len(items) + offset)
            items += extra
    else:
        # On resume, the items come from the session, but the row context is needed for new cells.
        items, row = detect(labels, settings.voxel_size, cell_type, comparison_compact, issue_rows,
                            table_ids.get("original"), species, reference_tag=tag,
                            source_labels=table_ids.get("source_labels"))

    def propose(anchor):
        return missing_proposal(anchor, labels, cell_voxels(labels), comparison_compact, row, settings.voxel_size)[0]

    def redetect(current, label_ids):
        return detect(current, settings.voxel_size, cell_type, comparison_compact, None, None, species,
                      label_ids=label_ids, reference_tag=tag)[0]

    if resume:
        controller = Controller.resume(image, labels, session, settings, propose, redetect)
    else:
        controller = Controller.create(image, labels, items, session, settings, propose, redetect)
    return Correction(
        controller, crop_name, original,
        output_path=os.path.join(output_folder, f"{crop_name}_annotations.tif"),
        table_path=os.path.join(output_folder, f"{crop_name}_changes.tsv"),
        comparison=comparison,
        comparison_name=f"{comparison_name} (comparison)" if comparison_name else None,
    )


def correct_annotation() -> None:
    parser = argparse.ArgumentParser(
        description="Correct the segmentation of an IHC or SGN crop in napari, cell by cell. The S3 segmentation "
        "of the crop is the start; with --edit_annotation, the annotation is. The tool finds the cells to correct, "
        "shows each with its neighbours, and fills the cell from a few edited slices. When all items are done, it "
        "writes <crop>_annotations.tif and <crop>_changes.tsv to the output folder. The session folder "
        "<crop>_session keeps the progress; start the same command again to resume. Work in progress: ready for "
        "tests on IHC crops; the options can change."
    )
    parser.add_argument("-i", "--image", required=True, help="The image tif of the crop.")
    parser.add_argument("-a", "--annotation", help="The annotation tif, shown as the comparison.")
    parser.add_argument("--edit_annotation", action="store_true",
                        help="Correct the annotation (-a) instead of the S3 segmentation.")
    parser.add_argument("-o", "--output_folder", required=True)
    parser.add_argument("--cell_type", choices=("ihc", "sgn"), help="The default comes from the stain in the name.")
    parser.add_argument("--reference", default="auto",
                        help="The S3 segmentation: 'auto' (IHC_v11b, IHC_v11, IHC_v10, IHC_v9 or SGN_v3, SGN_v2, "
                        "the first that exists), 'none', or a comma-separated list of sources.")
    parser.add_argument("--issues", help="A table of known issues (crop, label_id, z, y, x, reason, comment).")
    parser.add_argument("--cochlea", help="The cochlea, if the crop name does not give it.")
    parser.add_argument("--center", type=float, nargs=3, metavar=("X", "Y", "Z"),
                        help="The crop center in micrometer, if the crop name does not give it.")
    parser.add_argument("--contrast_limits", type=float, nargs=2, default=(0.0, 3000.0), metavar=("LOW", "HIGH"),
                        help="The contrast limits of the image at the start. Default: 0 3000.")
    parser.add_argument("--region", type=int, nargs=2, action="append", metavar=("A", "B"),
                        help="Segment the IHC row stretch between the cells A and B again (ids of the annotation, or "
                        "of the base without -a). The items are the proposed cells and A and B. Repeat for more "
                        "stretches.")
    parser.add_argument("--region_sources", default="all",
                        help="The S3 segmentations for --region: 'all' IHC model segmentations of the cochlea, or a "
                        "comma-separated list. Each is used as it is and with the IHC row correction.")
    parser.add_argument("--restart", action="store_true", help="Discard an existing session of the crop.")
    parser.add_argument("--remove_session", action="store_true",
                        help="Remove the session folder after the export. By default it stays as the record of the "
                        "edits.")
    defaults = Settings()
    group = parser.add_argument_group("settings", "The settings of the loop and the fill, see doc/annotation.md.")
    for setting in fields(Settings):
        default = getattr(defaults, setting.name)
        group.add_argument(f"--{setting.name}", type=type(default), default=default, help=f"Default: {default}.")
    args = parser.parse_args()

    from .widget import run_correction

    settings = Settings(**{setting.name: getattr(args, setting.name) for setting in fields(Settings)})
    region_sources = args.region_sources if args.region_sources == "all" else reference_list(args.region_sources)
    correction = prepare_correction(
        args.image, args.output_folder, args.annotation, args.edit_annotation, args.cell_type, args.reference,
        args.issues, args.cochlea, args.center, args.restart, settings, args.region, region_sources,
    )
    print(f"{correction.crop_name}: {len(correction.controller.queue)} items in the queue.")
    run_correction(correction.controller, correction.crop_name, correction.original, correction.output_path,
                   correction.table_path, correction.comparison, correction.comparison_name or "comparison",
                   args.remove_session, contrast_limits=tuple(args.contrast_limits))
