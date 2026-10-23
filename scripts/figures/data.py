"""Loaders for the figure data: S3 segmentation tables, cluster measurement tables and their caches.

Every loader keeps a pickle cache, because the S3 and cluster reads are slow. A loader reads the
cache when it exists, unless force_download is set.
"""
import json
import os
import pickle
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from flamingo_tools.postprocessing.sgn_subtype_utils import COCHLEAE as SUBTYPE_COCHLEAE, stain_to_type
from flamingo_tools.s3_utils import BUCKET_NAME, create_s3_target, read_table

from util import (
    SYNAPSE_DIR_ROOT, cochlea_components, cochleae_for, cohort_cochleae, length_column,
)

MEASUREMENT_ROOT = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/mobie_project/cochlea-lightsheet/tables"  # noqa
CHREEF_INTENSITY_ROOT = os.path.join(MEASUREMENT_ROOT, "measurements2")
LAVISION_INTENSITY_ROOT = os.path.join(MEASUREMENT_ROOT, "LaVision-OTOF")

# The LaVision OTOF cochleae. These are not in util.COCHLEA_DICT.
LAVISION_COCHLEAE_DICT = {
    "LaVision-OTOF23R": {"alias": "LV_M_03", "component": [4, 18, 7], "color": "#9C5027"},
    "LaVision-OTOF25R": {"alias": "LV_M_04", "component": [1], "color": "#67279C"},
}


def _read_cache(cache_path: str, force_download: bool):
    if force_download or not os.path.exists(cache_path):
        return None
    with open(cache_path, "rb") as f:
        return pickle.load(f)


def _write_cache(cache_path: str, data):
    with open(cache_path, "wb") as f:
        pickle.dump(data, f)


def source_table_dir(s3, cochlea: str, source_name: str) -> Optional[str]:
    """Get the S3 table folder of one segmentation source from the MoBIE dataset.json.

    Args:
        s3: S3 file system, as returned by create_s3_target.
        cochlea: Name of the cochlea.
        source_name: Name of the segmentation source, e.g. "SGN_v2".

    Returns:
        The table folder relative to the S3 root, or None if the cochlea has no such source.
    """
    content = s3.open(f"{BUCKET_NAME}/{cochlea}/dataset.json", mode="r", encoding="utf-8")
    sources = json.loads(content.read())["sources"]
    if source_name not in sources:
        return None
    rel_path = sources[source_name]["segmentation"]["tableData"]["tsv"]["relativePath"]
    return os.path.join(BUCKET_NAME, cochlea, rel_path)


def read_source_table(s3, cochlea: str, source_name: str, table_name: str = "default.tsv") -> pd.DataFrame:
    """Read one table of a segmentation source from S3.

    Args:
        s3: S3 file system, as returned by create_s3_target.
        cochlea: Name of the cochlea.
        source_name: Name of the segmentation source, e.g. "SGN_v2".
        table_name: File name of the table in the table folder of the source.

    Returns:
        The table.
    """
    table_dir = source_table_dir(s3, cochlea, source_name)
    if table_dir is None:
        raise KeyError(f"Cochlea {cochlea} has no source {source_name}.")
    return pd.read_csv(s3.open(os.path.join(table_dir, table_name), mode="rb"), sep="\t")


def get_tonotopic_data(
    cochleae: Optional[List[str]] = None,
    source_name: str = "IHC_v11",
    synapse_dir: str = SYNAPSE_DIR_ROOT,
    cache_path: Optional[str] = None,
    force_download: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Load per-cochlea tonotopic data from S3 and local synapse tables.

    Args:
        cochleae: Names of the cochleae to load. Defaults to the iDISCO cochleae. The component
            labels to keep are read from util.COCHLEA_DICT for source_name.
        source_name: IHC segmentation source name used as the S3 key (e.g. "IHC_v4c", "IHC_v9").
        synapse_dir: Root directory containing the ihc_counts_<version> subdirectories.
        cache_path: Path for the pickle cache. Defaults to "./tonotopic_data_<source_name>.pkl".
        force_download: Ignore the cached pickle and fetch the tables again.

    Returns:
        Dict mapping cochlea name to a DataFrame with columns
        label_id, length[µm], length_fraction, frequency[kHz], syn_per_IHC.
    """
    if cache_path is None:
        cache_path = f"./tonotopic_data_{source_name}.pkl"
    cached = _read_cache(cache_path, force_download)
    if cached is not None:
        return cached

    ihc_version = source_name.split("_")[1]
    cochleae = cohort_cochleae("idisco") if cochleae is None else list(cochleae)

    s3 = create_s3_target()
    tonotopic_data = {}
    for cochlea in cochleae:
        print("Processing cochlea:", cochlea)
        table = read_source_table(s3, cochlea, source_name)

        component_labels = cochlea_components(cochlea, "IHC", source_name)
        print(cochlea, component_labels)
        table = table[table.component_labels.isin(component_labels)]
        ihc_dir = f"ihc_counts_{ihc_version}"
        tab_path = os.path.join(synapse_dir, ihc_dir, f"ihc_count_{cochlea}.tsv")
        syn_tab = pd.read_csv(tab_path, sep="\t")
        syn_ids = syn_tab["label_id"].values

        syn_per_ihc = [0 for _ in range(len(table))]
        table.loc[:, "syn_per_IHC"] = syn_per_ihc
        for syn_id in syn_ids:
            table.loc[table["label_id"] == syn_id, "syn_per_IHC"] = syn_tab.at[syn_tab.index[syn_tab["label_id"] == syn_id][0], "synapse_count"]  # noqa

        try:
            values = table[["label_id", "length[µm]", "length_fraction", "frequency[kHz]", "syn_per_IHC"]]
        except KeyError:
            print("Could not find the values for", cochlea, "it will be skipped.")
            continue

        tonotopic_data[cochlea] = values

    _write_cache(cache_path, tonotopic_data)
    return tonotopic_data


def load_ribbon_synapse_counts(
    synapse_dir: Optional[str] = None,
    ihc_version: str = "v11",
    cochleae: Optional[List[str]] = None,
    exclude_zero: bool = False,
    require_all: bool = False,
) -> list:
    """Load and pool per-IHC synapse counts for the requested cochleae.

    ``exclude_zero`` is opt-in because zero-detection IHCs are part of the measurement in the
    other manuscript figures. Figure 5 uses it for the gerbil staining-quality sensitivity
    analysis. ``require_all`` prevents that panel from silently plotting a partial cohort when
    one of its expected tables has not been generated yet.

    Args:
        synapse_dir: Directory with the per-IHC synapse count tables. Defaults to the
            ihc_counts_<ihc_version> directory below util.SYNAPSE_DIR_ROOT.
        ihc_version: IHC segmentation version of the default synapse_dir.
        cochleae: Cochleae to pool. Defaults to the iDISCO cochleae.
        exclude_zero: Leave out the IHCs without a synapse.
        require_all: Raise if a table of one of the cochleae is missing.

    Returns:
        The synapse count of every IHC of the cochleae.
    """
    if synapse_dir is None:
        synapse_dir = os.path.join(SYNAPSE_DIR_ROOT, f"ihc_counts_{ihc_version}")
    if cochleae is None:
        cochleae = cohort_cochleae("idisco")

    entries = [entry for entry in os.scandir(synapse_dir) if entry.is_file()]
    tables = []
    missing = []
    for cochlea in cochleae:
        matches = [entry.path for entry in entries if cochlea in entry.name]
        if len(matches) > 1:
            raise ValueError(f"Found multiple synapse-count tables for {cochlea}: {matches}")
        if matches:
            tables.extend(matches)
        else:
            missing.append(cochlea)
    if require_all and missing:
        raise FileNotFoundError(
            f"Missing synapse-count tables for {missing} in {synapse_dir}. "
            "Refusing to plot an incomplete cohort."
        )

    syn_counts = []
    for tab in tables:
        x = pd.read_csv(tab, sep="\t")
        counts = x["synapse_count"]
        if exclude_zero:
            counts = counts[counts > 0]
        syn_counts.extend(counts.values.tolist())
    return syn_counts


def get_chreef_data(
    cache_path: str = "./chreef_data.pkl",
    force_download: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Load the SGN_v2 tables of the ChReef mouse cochleae, merged with their GFP intensities.

    The merge is an inner join on label_id, so an SGN without an intensity measurement is left out.

    Args:
        cache_path: Path for the pickle cache.
        force_download: Ignore the cached pickle and fetch the tables again.

    Returns:
        Mapping of cochlea name to a table with label_id, length[µm], frequency[kHz],
        marker_labels and the columns of the object measures table.
    """
    cached = _read_cache(cache_path, force_download)
    if cached is not None:
        return cached

    source_name = "SGN_v2"
    cochleae_dict = cochleae_for(cohort_cochleae("chreef_mouse"), "SGN", source_name)
    s3 = create_s3_target()
    chreef_data = {}
    for cochlea, meta in cochleae_dict.items():
        print("Processing cochlea:", cochlea)
        table = read_source_table(s3, cochlea, source_name)
        component_labels = meta["component"]
        print(cochlea, component_labels)
        table = table[table.component_labels.isin(component_labels)]
        try:
            values = table[["label_id", "length[µm]", "frequency[kHz]", "marker_labels"]]
        except KeyError:
            print("Could not find the values for", cochlea, "it will be skipped.")
            continue

        fname = f"{cochlea.replace('_', '-')}_GFP_SGN-v2_object-measures.tsv"
        intensity_file = os.path.join(CHREEF_INTENSITY_ROOT, fname)
        assert os.path.exists(intensity_file), intensity_file
        intensity_table = pd.read_csv(intensity_file, sep="\t")
        values = values.merge(intensity_table, on="label_id")

        chreef_data[cochlea] = values

    _write_cache(cache_path, chreef_data)
    return chreef_data


def get_otof_data(
    cache_path: str = "./otof_data.pkl",
    force_download: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Load the IHC_v11 tables of the OTOF gene-therapy cochleae.

    Args:
        cache_path: Path for the pickle cache.
        force_download: Ignore the cached pickle and fetch the tables again.

    Returns:
        Mapping of cochlea name to a table with label_id, length[µm], frequency[kHz] and
        marker_labels.
    """
    cached = _read_cache(cache_path, force_download)
    if cached is not None:
        return cached

    source_name = "IHC_v11"
    s3 = create_s3_target()
    otof_data = {}
    for cochlea, meta in cochleae_for(cohort_cochleae("otof_mouse"), "IHC", source_name).items():
        print("Processing cochlea:", cochlea)
        table = read_source_table(s3, cochlea, source_name)
        component_labels = meta["component"]
        print(cochlea, component_labels)
        table = table[table.component_labels.isin(component_labels)]
        try:
            values = table[["label_id", "length[µm]", "frequency[kHz]", "marker_labels"]]
        except KeyError:
            print("Could not find the values for", cochlea, "it will be skipped.")
            continue

        otof_data[cochlea] = values

    _write_cache(cache_path, otof_data)
    return otof_data


def get_lavision_otof_data(
    cache_path: str = "./otof_data_LaVision.pkl",
    force_download: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Load the IHC_LOWRES-v3 tables of the LaVision OTOF cochleae, merged with their intensities.

    The merge is an inner join on label_id, so an IHC without an intensity measurement is left out.

    Args:
        cache_path: Path for the pickle cache.
        force_download: Ignore the cached pickle and fetch the tables again.

    Returns:
        Mapping of cochlea name to a table with label_id, length[µm], frequency[kHz],
        frequency-mueller[kHz], expression_classification and the columns of the object measures
        table.
    """
    cached = _read_cache(cache_path, force_download)
    if cached is not None:
        return cached

    source_name = "IHC_LOWRES-v3"
    s3 = create_s3_target()
    otof_data = {}
    for cochlea, meta in LAVISION_COCHLEAE_DICT.items():
        print("Processing cochlea:", cochlea)
        table = read_source_table(s3, cochlea, source_name)
        component_labels = meta["component"]
        print(cochlea, component_labels)
        table = table[table.component_labels.isin(component_labels)]
        try:
            values = table[["label_id", "length[µm]", "frequency[kHz]", "frequency-mueller[kHz]",
                            "expression_classification"]]
        except KeyError:
            print("Could not find the values for", cochlea, "it will be skipped.")
            continue

        fname = f"{cochlea.replace('_', '-')}_rbOtof_IHC-LOWRES-v3_object-measures.tsv"
        intensity_file = os.path.join(LAVISION_INTENSITY_ROOT, fname)
        assert os.path.exists(intensity_file), intensity_file
        intensity_table = pd.read_csv(intensity_file, sep="\t")
        values = values.merge(intensity_table, on="label_id")

        otof_data[cochlea] = values

    _write_cache(cache_path, otof_data)
    return otof_data


def _match_column(columns, prefix: str, exclude: Optional[str] = None) -> str:
    """Find the single column starting with `prefix`, tolerating encoding-mangled headers."""
    candidates = [c for c in columns if c.lower().startswith(prefix) and c != exclude]
    assert len(candidates) == 1, f"Expected one column starting with {prefix!r}, found {candidates}"
    return candidates[0]


def get_gerbil_fchrimson_data(
    cache_path: str = "./gerbil_fchrimson_sgn_data.pkl",
    force_download: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Load the GFP-marked SGN_v2 tables of the f-Chrimson gerbil cochleae.

    Args:
        cache_path: Path for the pickle cache.
        force_download: Ignore the cached pickle and fetch the tables from S3 again.

    Returns:
        Mapping of cochlea name to a table with columns label_id, frequency[kHz], marker_labels.
    """
    cached = _read_cache(cache_path, force_download)
    if cached is not None:
        return cached

    source_name = "SGN_v2"
    s3 = create_s3_target()
    fchrimson_data = {}
    for cochlea, meta in cochleae_for(cohort_cochleae("fchrimson_gerbil"), "SGN", source_name).items():
        print("Processing cochlea:", cochlea)
        table = read_source_table(s3, cochlea, source_name, table_name="GFP_SGN-v2.tsv")
        table = table[table.component_labels.isin(meta["component"])]

        freq_col = _match_column(table.columns, "frequency")
        values = table[["label_id", freq_col, "marker_labels"]].rename(columns={freq_col: "frequency[kHz]"})

        marker_labels = values["marker_labels"].to_numpy()
        if not np.isin(marker_labels, [1, 2]).any():
            # Non-injected cochleae are sometimes left fully unannotated (0) instead of being
            # individually marked negative (2). A fully non-injected cochlea is entirely
            # marker-negative, so treat this case as all-negative rather than leaving a 0/0
            # efficiency downstream.
            values = values.assign(marker_labels=2)

        fchrimson_data[cochlea] = values

    _write_cache(cache_path, fchrimson_data)
    return fchrimson_data


def get_gerbil_density_data(
    cochleae: Optional[List[str]] = None,
    cache_path: str = "./gerbil_fchrimson_density_data.pkl",
    force_download: bool = False,
) -> Dict[str, dict]:
    """Load the SGN density files of gerbil cochleae.

    The extended density file is preferred, because it holds several positions per cochlear
    region. Cochleae without it fall back to the preset file with one position per region.

    Args:
        cochleae: Cochleae to load. Defaults to the f-Chrimson and the untreated gerbils, so that
            one cache serves the density panel and its reference band.
        cache_path: Path for the pickle cache.
        force_download: Ignore the cached pickle and fetch the density files from S3 again.

    Returns:
        Mapping of cochlea name to its parsed density dict, keyed by position.
    """
    if cochleae is None:
        cochleae = list(cohort_cochleae("fchrimson_gerbil")) + list(cohort_cochleae("wt_gerbil"))
    cached = _read_cache(cache_path, force_download)
    # A cache written for a smaller cohort would raise a KeyError downstream instead.
    if cached is not None and all(cochlea in cached for cochlea in cochleae):
        return cached

    s3 = create_s3_target()
    density_data = {}
    for cochlea in cochleae:
        table_dir = source_table_dir(s3, cochlea, "SGN_v2")
        density_path = os.path.join(table_dir, "SGN_density_2d_extended.json")
        if not s3.exists(density_path):
            density_path = os.path.join(table_dir, "SGN_density_2d.json")
        print(f"Processing cochlea: {cochlea}, density file: {os.path.basename(density_path)}")

        density_content = s3.open(density_path, mode="r", encoding="utf-8")
        density_data[cochlea] = json.loads(density_content.read())

    _write_cache(cache_path, density_data)
    return density_data


def get_sgn_length_data(
    cochleae: List[str],
    source_name: str = "SGN_v2",
    cache_path: Optional[str] = None,
    force_download: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Load the run length of every SGN of the given cochleae from the S3 segmentation tables.

    The cache holds every cochlea that was ever loaded. A cochlea that is not in the cache is
    fetched and added to it.

    Args:
        cochleae: Names of the cochleae to load.
        source_name: Name of the SGN segmentation source in the MoBIE dataset.
        cache_path: Path for the pickle cache. Defaults to "./sgn_length_data_<source_name>.pkl".
        force_download: Ignore the cached pickle and fetch the tables from S3 again.

    Returns:
        Mapping of cochlea name to a table with the columns label_id, length_fraction and
        length[µm].
    """
    if cache_path is None:
        cache_path = f"./sgn_length_data_{source_name}.pkl"

    cached = _read_cache(cache_path, force_download) or {}
    if all(cochlea in cached for cochlea in cochleae):
        return {cochlea: cached[cochlea] for cochlea in cochleae if cached[cochlea] is not None}

    s3 = create_s3_target()
    for cochlea in cochleae:
        if cochlea in cached:
            continue
        print("Processing cochlea:", cochlea)
        if source_table_dir(s3, cochlea, source_name) is None:
            print(f"Cochlea {cochlea} has no source {source_name}, it will be skipped.")
            cached[cochlea] = None
            continue
        table = read_source_table(s3, cochlea, source_name)

        # The tonotopic mapping sets the length fraction of instances outside of the mapped
        # components to 0. Without this filter they pile up in the first bin at the apex.
        table = table[table.component_labels.isin(cochlea_components(cochlea, "SGN", source_name))]

        try:
            length_col = length_column(table)
            values = table[["label_id", "length_fraction", length_col]]
        except KeyError:
            print(f"Cochlea {cochlea} has no run length columns, it will be skipped. "
                  "Run flamingo_tools.tonotopic_mapping for it first.")
            cached[cochlea] = None
            continue

        cached[cochlea] = values.rename(columns={length_col: "length[µm]"})

    _write_cache(cache_path, cached)
    return {cochlea: cached[cochlea] for cochlea in cochleae if cached[cochlea] is not None}


def get_subtype_data(cochleae: List[str]) -> Dict[str, dict]:
    """Load the SGN subtype classification and the frequencies of several cochleae from S3.

    Only SGNs with a positive or negative label for every subtype stain are kept.

    Args:
        cochleae: Keys of sgn_subtype_utils.COCHLEAE.

    Returns:
        Mapping of cochlea name to a dict with "classification", one "<subtype> (<stains>)" string
        per SGN, and "frequencies", the frequency of every SGN in kHz.
    """
    results = {}
    for cochlea in cochleae:
        meta = SUBTYPE_COCHLEAE[cochlea]
        seg_name = meta["output_seg"] if "output_seg" in meta else meta["seg_data"]
        component_list = meta["component_list"] if "component_list" in meta else [1]
        if "label_stains" in meta:
            stain_channels = meta["label_stains"]["subtype_label"]
        else:
            stain_channels = meta["subtype_stains"]

        table = read_table(f"{cochlea}/tables/{seg_name}/default.tsv", s3=True)
        table = table[table["component_labels"].isin(component_list)]

        # filter subtype table
        for chan in stain_channels:
            column = f"marker_{chan}"
            table = table.loc[table[column].isin([1, 2])]

        classification = []
        for chan in stain_channels:
            column = f"marker_{chan}"
            subset = table.loc[table[column].isin([1, 2])]
            marker = list(subset[column])
            chan_classification = []
            for m in marker:
                if m == 1:
                    chan_classification.append(f"{chan}+")
                elif m == 2:
                    chan_classification.append(f"{chan}-")
            classification.append(chan_classification)

        # Unify the classification and assign colors
        assert len(classification) in (1, 2)
        if len(classification) == 2:
            cls1, cls2 = classification[0], classification[1]
            assert len(cls1) == len(cls2)
            classification = [f"{c1} / {c2}" for c1, c2 in zip(cls1, cls2)]
        else:
            classification = classification[0]

        classification = [stain_to_type(cls) for cls in classification]
        classification = [f"{stype} ({stain})" for stype, stain in classification]

        freq = table["frequency[kHz]"].values
        assert len(freq) == len(classification)

        results[cochlea] = {"classification": classification, "frequencies": freq}
    return results
