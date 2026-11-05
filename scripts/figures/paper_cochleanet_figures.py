"""Create the plot panels of the CochleaNet paper.

The script writes every published plot panel and its legends into OUTPUT_DIR/fig<N>/ and
OUTPUT_DIR/supp_fig<N>/. The panel ids follow the published numbering, e.g. "2c" or "S4b".
Renderings and schemes are not made here. Panel 6D, Supp. Fig. 9D and the "Rankovic et al."
part of Supp. Fig. 9E come from outside this repository.

Panels that read cluster paths (/mnt/vast-nhr) or a cache that is not available fail on their
own. The script then writes the remaining panels, lists the failed ones and exits with 1.

Examples:
    python paper_cochleanet_figures.py ./panels
    python paper_cochleanet_figures.py ./panels -p 2 S4b --variants
"""
import argparse
import os
import re
import sys
import traceback

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import data
import gene_therapy
import literature_comparison
import plot_annotator_variance
import plot_sgn_density_profile
import segmentation_accuracy
import sgn_subtypes
import tonotopic_mapping
from util import (
    COCHLEA_DICT, SYNAPSE_DIR_ROOT, cochleae_for, documented_counts, literature_reference_values,
    plot_cochlea_legend,
)

FILE_EXTENSION = "png"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

# Per structure of panel 2C: the x-axis label, the annotator and network accuracy files, the
# released model that the panel plots, and the model set behind its error bar. Key order
# determines the left-to-right order on the x-axis.
#
# The SGN and IHC model sets are training-seed replicates. For synapses, the set holds the five
# highest-test-F1 v3-style models: the original v3 model and the four best seed variants. That
# spread is the spread among the selected best models, not an unbiased estimate over every seed
# run. All entries are evaluated at the same settings: SGN with component filtering, IHC with
# component filtering and without the zero-synapse filter, synapses with the assignment to the
# IHCs.
ACCURACY_MODEL_SETS = {
    "SGN_v2": {
        "label": "SGN",
        "accuracy_file": "SGN_3D.json",
        "consensus_file": "consensus_SGN.json",
        "released": "v2",
        "replicates": ["v2-1", "v2-2", "v2-3", "v2-4"],
    },
    "IHC_v11": {
        "label": "IHC",
        "accuracy_file": "IHC_3D.json",
        "consensus_file": "consensus_IHC.json",
        "released": "v11",
        "replicates": ["v11-1", "v11-2", "v11-3", "v11-4"],
    },
    "synapses_v3": {
        "label": "Synapse",
        "accuracy_file": "synapses.json",
        "consensus_file": "consensus_synapses.json",
        "released": "v3",
        "replicates": ["v3-4", "v3", "v3-3-best", "v3-6", "v3-5-epoch71"],
    },
}

# The methods of Supp. Fig. 3, keys of segmentation_accuracy.PLOT_METADATA.
METHODS_SGN = [
    "stardist", "micro-sam", "micro-sam_finetuned", "cellpose3", "cellpose3_finetuned", "cellpose-sam",
    "spiner2D", "distance_unet",
]
METHODS_IHC = [
    "micro-sam", "micro-sam_finetuned", "cellpose3", "cellpose3_finetuned", "cellpose-sam", "distance_unet_v11",
]
RUNTIME_METHODS_SGN = ["stardist", "micro-sam", "cellpose3", "cellpose-sam", "spiner2D", "distance_unet"]
RUNTIME_METHODS_IHC = ["micro-sam", "cellpose3", "cellpose-sam", "distance_unet_v11"]

# Fold keys per structure for the fold-variation variants. The outer key selects the accuracy
# file (<key>.json) and the x-axis position; the value lists the fold entries to average.
FOLD_KEYS = {
    "SGN": [f"distance_unet_f{fold}" for fold in range(5)],
    "IHC": [f"distance_unet_v11-f{fold}" for fold in range(5)],
    "synapses": [f"v5-f{fold}" for fold in range(5)],
}

# Training-seed variants, for the seed-variation variant. These are a different experiment from
# FOLD_KEYS above: the fold keys come from cross-validation on 2D slices, whereas these are full 3D
# segmentations of the validation cochleae by networks that share the training data and the split
# and differ only in the training seed. So this panel shows how much of the accuracy depends on the
# seed alone, and the two must not be conflated. The outer key selects the accuracy file.
SEED_KEYS = {
    "SGN_3D": ["v2-1", "v2-2", "v2-3", "v2-4"],
    "IHC_3D": ["v11-1", "v11-2", "v11-3", "v11-4"],
    # The synapse replicates share the v3 training data and split (random_state 42) and were
    # evaluated at the production settings, see scripts/validation/synapses/prediction.py.
    "synapses": ["v3-1", "v3-2", "v3-3", "v3-4"],
}
SEED_GROUP_LABELS = {"SGN_3D": "SGN", "IHC_3D": "IHC", "synapses": "Synapses"}

# The cochleae of the published panels. The panels read them from here and not from
# util.COHORT_DICT, so that a later change of a cohort does not change a published panel. The
# aliases, colors and component lists still come from util.COCHLEA_DICT.
PAPER_COHORTS = {
    # iDISCO reference mice: 2D, 3C, S4A and the untreated counts of 4C and 6C.
    "idisco": ["M_LR_000226_L", "M_LR_000226_R", "M_LR_000227_L", "M_LR_000227_R"],
    # ChReef mice: 4C-E and the alias and x-axis order of S6D. 4C-E leave out the M_LR_000143 pair.
    "chreef_mouse": [
        "M_LR_000143_L", "M_LR_000144_L", "M_LR_000145_L", "M_LR_000153_L", "M_LR_000155_L", "M_LR_000189_L",
        "M_LR_000143_R", "M_LR_000144_R", "M_LR_000145_R", "M_LR_000153_R", "M_LR_000155_R", "M_LR_000189_R",
    ],
    # OTOF gene therapy mice: 6C.
    "otof_mouse": ["M_AMD_OTOF27_L", "M_AMD_OTOF27_R", "M_AMD_OTOF28_L", "M_AMD_OTOF28_R"],
    # f-Chrimson gerbils: 5D, 5E and S8E.
    "fchrimson_gerbil": [
        "G_EK_000049_L", "G_EK_000049_R", "G_EK_000071_L", "G_EK_000071_R",
        "G_EK_000074_L", "G_EK_000074_R", "G_EK_000076_L", "G_EK_000076_R",
    ],
    # G_EK_000049 received the injection postnatally, the other three animals as adults.
    "fchrimson_gerbil_postnatal": ["G_EK_000049_L", "G_EK_000049_R"],
    # Wild-type gerbils: the SGN and IHC counts of 5C, the untreated band of 5D and S7D.
    # G_LR_000301_L has no tables on S3, so 5D, S7C and S7D need a cache that holds it.
    "wt_gerbil": ["G_EK_000233_L", "G_LR_000301_L", "G_LR_000301_R", "G_LR_000302_R"],
    # S7C leaves out G_EK_000233_L, which is incomplete along Rosenthal's canal.
    "wt_gerbil_normalized": ["G_LR_000301_L", "G_LR_000301_R", "G_LR_000302_R"],
    # The synapse axis of 5C leaves out G_LR_000301_L, because its staining-limited estimate stays
    # distinctly low even after removing zero-count IHCs. The other three cochleae keep that
    # zero-exclusion rule.
    "wt_gerbil_synapses": ["G_EK_000233_L", "G_LR_000301_R", "G_LR_000302_R"],
}

# Groups of SGN subtype cochleae (keys of sgn_subtype_utils.COCHLEAE) with a common staining. The
# key lists the subtypes the staining can distinguish, a suffix after "-" separates groups with the
# same subtypes. "Type I;Type II" also holds the Type I/II offsets of S4C.
SUBTYPE_COHORTS = {
    "Type Ia;Type Ib;Type Ic;Type II": ["M_LR_000098_L", "M_LR_N152_L"],  # "M_LR_N98_R"
    "Type Ia;Type Ib;Type Ic;Type II-noPV": ["M_AMD_N180_L", "M_AMD_N180_R", "M_AMD_N190_L", "M_AMD_N190_R"],
    "Type Ia;Type Ib;Type Ic;Type II-old+new": [
        "M_LR_000098_L", "M_LR_N152_L", "M_AMD_N180_L", "M_AMD_N180_R", "M_AMD_N190_L", "M_AMD_N190_R",
    ],
    "Type Ia;Type Ib;Type Ic;Type II-all": [
        "M_LR_000098_L", "M_LR_N98_R", "M_LR_N152_L", "M_AMD_N180_L", "M_AMD_N180_R", "M_AMD_N190_L", "M_AMD_N190_R",
    ],
    "Type I;Type II": ["M_LR_000184_L", "M_LR_000184_R", "M_LR_000260_L"],
    "Type Ib;Type Ic;inconclusive": ["M_LR_N110_L", "M_LR_N110_R", "M_LR_N152_R"],
}

IDISCO_DICT = cochleae_for(PAPER_COHORTS["idisco"], "IHC", "IHC_v11")
CHREEF_DICT = cochleae_for(PAPER_COHORTS["chreef_mouse"], "SGN", "SGN_v2")
OTOF_DICT = cochleae_for(PAPER_COHORTS["otof_mouse"], "IHC", "IHC_v11")
FCHRIMSON_DICT = cochleae_for(PAPER_COHORTS["fchrimson_gerbil"], "SGN", "SGN_v2")

# The adult injected gerbil cochleae, whose mean is the reference line of Supp. Fig. 8E. The
# postnatal animal is left out, because it is not comparable to them.
ADULT_INJECTED_GERBILS = [
    name for name in PAPER_COHORTS["fchrimson_gerbil"]
    if name.endswith("_L") and name not in PAPER_COHORTS["fchrimson_gerbil_postnatal"]
]

PANELS = {}
VARIANTS = {}

# Data that more than one panel reads, loaded once per run.
_LOADED = {}


def panel(*panel_ids, variant=False):
    """Register a function that writes the files of one or more panels."""
    def register(function):
        for panel_id in panel_ids:
            (VARIANTS if variant else PANELS)[panel_id] = function
        return function
    return register


def out(args, panel_id, suffix="", variant=False):
    """Get the output path of a panel file, e.g. <output_dir>/supp_fig4/supp_fig4b_legend.png.

    A variant goes to <output_dir>/variants/ instead of the folder of its figure.
    """
    match = re.fullmatch(r"(S?)(\d+)([a-z]?)", panel_id)
    figure = ("supp_fig" if match.group(1) else "fig") + match.group(2)
    folder = os.path.join(args.output_dir, "variants" if variant else figure)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f"{figure}{match.group(3)}{suffix}.{FILE_EXTENSION}")


def _load(name, function):
    if name not in _LOADED:
        _LOADED[name] = function()
    return _LOADED[name]


def _cache(args, file_name):
    return os.path.join(args.cache_dir, file_name)


def _tonotopic_data(args):
    return _load("tonotopic", lambda: data.get_tonotopic_data(
        cochleae=PAPER_COHORTS["idisco"], source_name="IHC_v11", synapse_dir=args.synapse_dir,
        cache_path=_cache(args, "tonotopic_data_IHC_v11.pkl"), force_download=args.refresh_cache))


def _subtype_data(grouping):
    return _load(f"subtypes {grouping}", lambda: data.get_subtype_data(SUBTYPE_COHORTS[grouping]))


def _chreef_data(args):
    def load():
        chreef_data = data.get_chreef_data(
            PAPER_COHORTS["chreef_mouse"], _cache(args, "chreef_data.pkl"), args.refresh_cache)
        # M_LR_000143_L is a complete outlier. Its right cochlea goes too, so that only pairs remain.
        return {name: table for name, table in chreef_data.items() if not name.startswith("M_LR_000143")}
    return _load("chreef", load)


def _otof_data(args):
    return _load("otof", lambda: data.get_otof_data(
        PAPER_COHORTS["otof_mouse"], _cache(args, "otof_data.pkl"), args.refresh_cache))


def _gerbil_sgn_data(args):
    return _load("gerbil sgn", lambda: data.get_gerbil_fchrimson_data(
        PAPER_COHORTS["fchrimson_gerbil"], _cache(args, "gerbil_fchrimson_sgn_data.pkl"), args.refresh_cache))


def _gerbil_density_data(args):
    return _load("gerbil density", lambda: data.get_gerbil_density_data(
        cochleae=PAPER_COHORTS["fchrimson_gerbil"] + PAPER_COHORTS["wt_gerbil"],
        cache_path=_cache(args, "gerbil_fchrimson_density_data.pkl"), force_download=args.refresh_cache))


def _sgn_length_data(args, cochleae):
    return data.get_sgn_length_data(
        cochleae, cache_path=_cache(args, "sgn_length_data_SGN_v2.pkl"), force_download=args.refresh_cache)


def _synapse_counts_dir(args):
    return os.path.join(args.synapse_dir, "ihc_counts_v11")


# The order of the calls inside a panel follows the old figure scripts. Some functions do not
# set the matplotlib style themselves, so their output depends on the calls before them.


@panel("1a")
def fig1a(args):
    if not args.napari:
        print("Panel 1a is a napari rendering. Use --napari to show it.")
        return
    tonotopic_mapping.show_analysis_overview()


@panel("2c")
def fig2c(args):
    segmentation_accuracy.plot_accuracy_manual_vs_automatic(
        out(args, "2c"), args.data_dir, ACCURACY_MODEL_SETS, plot=args.plot, error_bars="production")
    segmentation_accuracy.plot_accuracy_legend(out(args, "2c", "_legend_shapes"), plot_mode="shapes")
    segmentation_accuracy.plot_accuracy_legend(out(args, "2c", "_legend_colors"), plot_mode="colors")


@panel("2c", variant=True)
def fig2c_variants(args):
    for keyword, prefix in (("all", ""), ("pairwise", "_pairwise")):
        for error_bars, suffix in (("none", "_no_sd"), ("production", "_sd_production"), ("mean", "_sd_mean")):
            if keyword == "all" and error_bars == "production":
                continue  # The published panel.
            segmentation_accuracy.plot_accuracy_manual_vs_automatic(
                out(args, "2c", prefix + suffix, variant=True), args.data_dir, ACCURACY_MODEL_SETS,
                plot=args.plot, annotator_keyword=keyword, error_bars=error_bars)


@panel("2d")
def fig2d(args):
    cochleae = PAPER_COHORTS["idisco"]
    literature_comparison.plot_counts_vs_literature(
        out(args, "2d"),
        [
            {"values": documented_counts(cochleae, "SGN", "SGN_v2"), "label": "SGN",
             "literature": literature_reference_values("SGN"), "ylim": (8500, 12500),
             "yticks": list(range(9000, 12001, 1000)), "literature_label": True, "fill_color": "C0"},
            {"values": documented_counts(cochleae, "IHC", "IHC_v11"), "label": "IHC",
             "literature": literature_reference_values("IHC"), "ylim": (600, 800),
             "yticks": list(range(600, 801, 100))},
            {"values": data.load_ribbon_synapse_counts(_synapse_counts_dir(args), cochleae=cochleae),
             "label": "Synapses per IHC",
             "literature": literature_reference_values("synapse"), "ylim": (-1, 41),
             "yticks": [0, 10, 20, 30, 40, 50]},
        ],
        figsize=(10, 4.5), tick_size=16, plot=args.plot,
    )


@panel("3a")
def fig3a(args):
    tonotopic_mapping.plot_frequency_colorbar(
        out(args, "3a", "_colorbar"), plot=args.plot, cmap="plasma", dark_mode=True)
    if args.napari:
        tonotopic_mapping.show_tonotopic_mapping(cmap="plasma")


@panel("3c")
def fig3c(args):
    tonotopic_mapping.plot_synapses_per_band(
        _tonotopic_data(args), out(args, "3c"), plot=args.plot, use_alias=args.use_alias, trendline=True,
        cochleae_dict=IDISCO_DICT)
    plot_cochlea_legend(out(args, "3c", "_legend"), IDISCO_DICT, ncol=1)


@panel("3d")
def fig3d(args):
    grouping = "Type I;Type II"
    sgn_subtypes.plot_subtype_fraction(_subtype_data(grouping), out(args, "3d"), plot=args.plot)
    sgn_subtypes.plot_subtype_legend(out(args, "3d", "_legend"), grouping=grouping, ncol=1)


@panel("3e")
def fig3e(args):
    grouping = "Type Ia;Type Ib;Type Ic;Type II"
    sgn_subtypes.plot_subtype_fraction(_subtype_data(grouping), out(args, "3e"), plot=args.plot)
    sgn_subtypes.plot_subtype_legend(out(args, "3e", "_legend"), grouping=grouping, ncol=4,
                                     outlined_subtypes=["Type Ib", "Type Ic"])


@panel("3e", variant=True)
def fig3e_variants(args):
    for suffix in ("noPV", "old+new", "all"):
        grouping = f"Type Ia;Type Ib;Type Ic;Type II-{suffix}"
        sgn_subtypes.plot_subtype_fraction(
            _subtype_data(grouping), out(args, "3e", f"_{suffix}", variant=True), plot=args.plot)


@panel("3f")
def fig3f(args):
    sgn_subtypes.plot_subtype_per_band(
        _subtype_data("Type Ia;Type Ib;Type Ic;Type II"), out(args, "3f"), plot=args.plot, combine_IbIc=True)
    sgn_subtypes.plot_subtype_legend(out(args, "3f", "_legend"), grouping="Type Ia;Type Ib/Ic;Type II")


@panel("3f", variant=True)
def fig3f_variants(args):
    for suffix in ("noPV", "old+new", "all"):
        grouping = f"Type Ia;Type Ib;Type Ic;Type II-{suffix}"
        sgn_subtypes.plot_subtype_per_band(
            _subtype_data(grouping), out(args, "3f", f"_{suffix}", variant=True), plot=args.plot,
            combine_IbIc=True)
    sgn_subtypes.plot_subtype_legend(out(args, "3f", "_noPV_legend", variant=True),
                                     grouping="Type Ia;Type Ib;Type Ic;Type II")


@panel("4c")
def fig4c(args):
    gene_therapy.plot_count_vs_untreated(
        _chreef_data(args), out(args, "4c"), CHREEF_DICT,
        documented_counts(PAPER_COHORTS["idisco"], "SGN", "SGN_v2"), plot=args.plot, use_alias=args.use_alias,
        ylim=(5000, 14000), y_ticks=list(range(6000, 13000, 2000)))


@panel("4d")
def fig4d(args):
    gene_therapy.plot_expression_efficiency(
        _chreef_data(args), out(args, "4d"), CHREEF_DICT, plot=args.plot,
        use_alias=args.use_alias, ylim=(0.65, 1.05), y_ticks=np.arange(0.7, 1, 0.1))


def _plot_chreef_efficiency_per_band(args, save_path, trendline_std):
    gene_therapy.plot_efficiency_per_band(
        _chreef_data(args), save_path, CHREEF_DICT, animal="mouse", plot=args.plot,
        use_alias=args.use_alias, trendlines=True, trendline_std=trendline_std,
        trendline_colors=("gray", "gray"), ylim=(0.45, 1.05), y_ticks=np.arange(0.5, 1.01, 0.1))


@panel("4e")
def fig4e(args):
    gene_therapy.plot_pair_legend(_chreef_data(args), out(args, "4e", "_legend"), CHREEF_DICT,
                                  use_alias=args.use_alias)
    gene_therapy.plot_side_trendline_legend(out(args, "4e", "_legend_trendline"))
    _plot_chreef_efficiency_per_band(args, out(args, "4e"), trendline_std=False)


@panel("4e", variant=True)
def fig4e_variants(args):
    _plot_chreef_efficiency_per_band(args, out(args, "4e", "_std", variant=True), trendline_std=True)


@panel("5c")
def fig5c(args):
    literature_comparison.plot_counts_vs_literature(
        out(args, "5c"),
        [
            {"values": documented_counts(PAPER_COHORTS["wt_gerbil"], "SGN", "SGN_v2"), "label": "SGN",
             "literature": literature_reference_values("SGN", animal="gerbil"), "ylim": (14000, 30000),
             "yticks": list(range(16000, 30001, 4000)), "literature_label": True},
            {"values": documented_counts(PAPER_COHORTS["wt_gerbil"], "IHC", "IHC_v11"), "label": "IHC",
             "literature": literature_reference_values("IHC", animal="gerbil"), "ylim": (900, 1200),
             "yticks": list(range(900, 1201, 100))},
            {"values": data.load_ribbon_synapse_counts(
                _synapse_counts_dir(args), cochleae=PAPER_COHORTS["wt_gerbil_synapses"], exclude_zero=True,
                require_all=True),
             "label": "Synapses per IHC", "literature": literature_reference_values("synapse", animal="gerbil"),
             "ylim": (-1, 80), "yticks": list(range(0, 81, 20)), "box_zorder": None},
        ],
        figsize=(8.5, 4.5), tick_size=20, plot=args.plot,
    )


def _plot_gerbil_density(args, save_path, show_std):
    density_data = _gerbil_density_data(args)
    gene_therapy.plot_density_by_region(
        {name: density_data[name] for name in PAPER_COHORTS["fchrimson_gerbil"]}, save_path,
        cochleae_dict=FCHRIMSON_DICT, plot=args.plot, use_alias=args.use_alias, show_std=show_std,
        reference_values=gene_therapy.untreated_reference_values(
            {name: density_data[name] for name in PAPER_COHORTS["wt_gerbil"]}))


@panel("5d")
def fig5d(args):
    gene_therapy.plot_pair_legend(_gerbil_sgn_data(args), out(args, "5d", "_legend"),
                                  FCHRIMSON_DICT, use_alias=args.use_alias)
    _plot_gerbil_density(args, out(args, "5d"), show_std=True)


@panel("5d", variant=True)
def fig5d_variants(args):
    _plot_gerbil_density(args, out(args, "5d", "_no_sd", variant=True), show_std=False)
    gene_therapy.plot_count_vs_untreated(
        _sgn_length_data(args, PAPER_COHORTS["fchrimson_gerbil"]),
        out(args, "5", "_fchrimson_sgn_count", variant=True),
        FCHRIMSON_DICT, documented_counts(PAPER_COHORTS["wt_gerbil"], "SGN", "SGN_v2"),
        plot=args.plot, use_alias=args.use_alias)


@panel("5e")
def fig5e(args):
    gene_therapy.plot_expression_efficiency(
        _gerbil_sgn_data(args), out(args, "5e"), FCHRIMSON_DICT, plot=args.plot,
        use_alias=args.use_alias)


@panel("6c")
def fig6c(args):
    otof_data = _otof_data(args)
    gene_therapy.plot_count_vs_untreated(
        otof_data, out(args, "6c", "_ihc_count"), OTOF_DICT,
        documented_counts(PAPER_COHORTS["idisco"], "IHC", "IHC_v11"), plot=args.plot, use_alias=args.use_alias,
        count_label="IHC count per cochlea", xtick_labels=("Injected", "Non-Injected"),
        ylim=(500, 750), y_ticks=list(range(500, 800, 50)))
    gene_therapy.plot_expression_efficiency(
        otof_data, out(args, "6c", "_expression"), OTOF_DICT, plot=args.plot,
        use_alias=args.use_alias, xtick_labels=("Injected", "Non-Injected"), ylim=(-0.03, 0.4),
        y_ticks=[0.0, 0.1, 0.2, 0.3])
    gene_therapy.plot_pair_legend(otof_data, out(args, "6c", "_legend"), OTOF_DICT,
                                  use_alias=args.use_alias)


@panel("S3a")
def supp_fig3a(args):
    segmentation_accuracy.plot_method_comparison_legend(out(args, "S3", "_legend"))
    segmentation_accuracy.plot_method_comparison(
        out(args, "S3a"), plot=args.plot, segm="SGN", key_list=METHODS_SGN, data_dir=args.data_dir)


@panel("S3a", variant=True)
def supp_fig3_variants(args):
    def path(suffix):
        return out(args, "S3", suffix, variant=True)

    data_dir = args.data_dir
    segmentation_accuracy.plot_method_comparison(
        path("a_fold_sd"), plot=args.plot, segm="SGN", key_list=METHODS_SGN, data_dir=data_dir,
        ref_dict={"distance_unet": FOLD_KEYS["SGN"]})
    segmentation_accuracy.plot_method_comparison(
        path("c_fold_sd"), plot=args.plot, segm="IHC", key_list=METHODS_IHC, data_dir=data_dir,
        ref_dict={"distance_unet_v11": FOLD_KEYS["IHC"]})
    segmentation_accuracy.plot_method_comparison(
        path("_ihc_versions"), plot=args.plot, segm="IHC_3D", show_legend=True, data_dir=data_dir,
        key_list=["v9", "v10", "v11"], ylim=[0.80, 1.0])
    segmentation_accuracy.plot_fold_accuracy(path("_sgn_folds"), data_dir=data_dir, plot=args.plot)
    segmentation_accuracy.plot_method_comparison(
        path("_synapse_thresholds"), plot=args.plot, segm="synapses", data_dir=data_dir, show_legend=True,
        ylim=[0.5, 1.05], key_list=["v3", "v5", "v3_05t", "v5_f1val_threshold", "v5_train_threshold", "v5_05t"])
    for segm, name in (("SGN", "sgn"), ("IHC", "ihc"), ("synapses", "synapse")):
        segmentation_accuracy.plot_method_comparison(
            path(f"_{name}_fold_accuracy"), plot=args.plot, segm=segm, data_dir=data_dir, show_legend=True,
            ylim=[0.5, 1.05], key_list=FOLD_KEYS[segm])
    segmentation_accuracy.plot_replicate_variation(
        path("_fold_variation"), replicate_dict=FOLD_KEYS, data_dir=data_dir, plot=args.plot)
    segmentation_accuracy.plot_replicate_variation(
        path("_seed_variation"), replicate_dict=SEED_KEYS, data_dir=data_dir, group_labels=SEED_GROUP_LABELS,
        plot=args.plot)
    segmentation_accuracy.plot_method_comparison(
        path("_ihc_3d_versions"), plot=args.plot, segm="IHC_3D", data_dir=data_dir, show_legend=True,
        ylim=[0.8, 0.95])


@panel("S3b")
def supp_fig3b(args):
    segmentation_accuracy.plot_method_comparison(
        out(args, "S3b"), plot=args.plot, segm="SGN", mode="runtime", key_list=RUNTIME_METHODS_SGN,
        data_dir=args.data_dir)


@panel("S3c")
def supp_fig3c(args):
    segmentation_accuracy.plot_method_comparison(
        out(args, "S3c"), plot=args.plot, segm="IHC", key_list=METHODS_IHC, data_dir=args.data_dir)


@panel("S3d")
def supp_fig3d(args):
    segmentation_accuracy.plot_method_comparison(
        out(args, "S3d"), plot=args.plot, segm="IHC", mode="runtime", key_list=RUNTIME_METHODS_IHC,
        data_dir=args.data_dir)


@panel("S4a")
def supp_fig4a(args):
    tonotopic_mapping.plot_synapses_along_cochlea(
        _tonotopic_data(args), out(args, "S4a"), use_alias=args.use_alias, plot=args.plot, n_bins=25,
        top_axis=True, trendline=True, cohort_dict=IDISCO_DICT, alpha=1)
    plot_cochlea_legend(out(args, "S4a", "_legend"), IDISCO_DICT, ncol=3, meyer=True)


@panel("S4b")
def supp_fig4b(args):
    grouping = "Type I;Type II"
    sgn_subtypes.plot_subtype_per_band(_subtype_data(grouping), out(args, "S4b"), plot=args.plot)
    sgn_subtypes.plot_subtype_legend(out(args, "S4b", "_legend"), grouping=grouping)


@panel("S4c")
def supp_fig4c(args):
    cochleae = SUBTYPE_COHORTS["Type I;Type II"]
    sgn_subtypes.plot_type_offset(out(args, "S4c", "_mean"), plot=args.plot, plot_type="mean", cochleae=cochleae)
    sgn_subtypes.plot_type_offset(out(args, "S4c", "_points"), plot=args.plot, plot_type="points",
                                  cochleae=cochleae)
    sgn_subtypes.plot_offset_legend(out(args, "S4c", "_legend"), cochleae=cochleae)


@panel("S4d", "S4e")
def supp_fig4de(args):
    sgn_subtypes.plot_threshold_legend(out(args, "S4d", "_legend"), stain="CR", ncol=1)
    sgn_subtypes.plot_threshold_legend(out(args, "S4e", "_legend"), stain="Ntng1", ncol=1)
    sgn_subtypes.plot_marker_threshold_histograms(
        {"CR": out(args, "S4d"), "Ntng1": out(args, "S4e")}, cochlea="M_LR_N152_L", plot=args.plot,
        sharex=True, title_type="generic")


@panel("S4f")
def supp_fig4f(args):
    grouping = "Type Ib;Type Ic;inconclusive"
    sgn_subtypes.plot_subtype_fraction(_subtype_data(grouping), out(args, "S4f"), plot=args.plot)
    sgn_subtypes.plot_subtype_legend(out(args, "S4f", "_legend"), grouping=grouping, ncol=1)


@panel("S4g")
def supp_fig4g(args):
    sgn_subtypes.plot_subtype_per_band(
        _subtype_data("Type Ib;Type Ic;inconclusive"), out(args, "S4g"), plot=args.plot, combine_IbIc=True)
    sgn_subtypes.plot_subtype_legend(out(args, "S4g", "_legend"), grouping="Type Ib/Ic;inconclusive", ncol=1)


@panel("S6d")
def supp_fig6d(args):
    if args.variance_dir is None:
        raise ValueError("Panel S6d needs --variance_dir, the folder with the *_variance.json files.")
    df = plot_annotator_variance.load_variance_records(
        args.variance_dir, marker="GFP", use_alias=args.use_alias, cochleae_dict=CHREEF_DICT)
    plot_annotator_variance.plot_annotator_variance(
        df, save_path=out(args, "S6d"), n_breaks=0, cochleae_dict=CHREEF_DICT, plot=args.plot)
    plot_annotator_variance.plot_annotator_variance_legend(df, save_path=out(args, "S6d", "_legend"))


def _plot_wt_gerbil_density(args, save_path, absolute_runlength):
    length_data = _sgn_length_data(args, PAPER_COHORTS["wt_gerbil" if absolute_runlength else "wt_gerbil_normalized"])
    metadata = plot_sgn_density_profile.build_plot_metadata(PAPER_COHORTS["wt_gerbil"], cohort="WT gerbil")
    plot_sgn_density_profile.plot_sgn_density_profile(
        length_data, save_path=save_path, mode="bins", n_bins=10, absolute_runlength=absolute_runlength,
        cochleae_dict=metadata, use_alias=args.use_alias, plot=args.plot,
        trendline=not absolute_runlength, trendline_std=False,
        trendline_label=None if absolute_runlength else "Mean", top_axis=not absolute_runlength,
        animal="gerbil", show_legend=False, length_info=False,
    )
    return metadata


@panel("S7c")
def supp_fig7c(args):
    metadata = _plot_wt_gerbil_density(args, out(args, "S7c"), absolute_runlength=False)
    plot_sgn_density_profile.export_wt_gerbil_legend(metadata, out(args, "S7", "_legend"), use_alias=args.use_alias)


@panel("S7d")
def supp_fig7d(args):
    _plot_wt_gerbil_density(args, out(args, "S7d"), absolute_runlength=True)


@panel("S8e")
def supp_fig8e(args):
    gerbil_data = _gerbil_sgn_data(args)
    gene_therapy.plot_pair_legend(gerbil_data, out(args, "S8e", "_legend"), FCHRIMSON_DICT,
                                  use_alias=args.use_alias)
    gene_therapy.plot_adult_mean_legend(out(args, "S8e", "_legend_adult_mean"))
    gene_therapy.plot_efficiency_per_band(
        gerbil_data, out(args, "S8e"), FCHRIMSON_DICT, animal="gerbil", plot=args.plot,
        use_alias=args.use_alias, reference_cochleae=ADULT_INJECTED_GERBILS)


@panel("S8e", variant=True)
def supp_fig8e_variants(args):
    gerbil_data = _gerbil_sgn_data(args)
    gene_therapy.plot_side_trendline_legend(out(args, "S8e", "_legend_trendline", variant=True))
    gene_therapy.plot_efficiency_per_band(
        gerbil_data, out(args, "S8e", "_std", variant=True), FCHRIMSON_DICT,
        animal="gerbil", plot=args.plot, use_alias=args.use_alias, trendline_std=True,
        reference_cochleae=ADULT_INJECTED_GERBILS)
    gene_therapy.plot_efficiency_per_band(
        gerbil_data, out(args, "S8e", "_postnatal", variant=True), FCHRIMSON_DICT,
        animal="gerbil", plot=args.plot, use_alias=args.use_alias, cochleae=PAPER_COHORTS["fchrimson_gerbil_postnatal"],
        color_by_side=True, reference_cochleae=ADULT_INJECTED_GERBILS)
    gene_therapy.plot_side_legend(
        out(args, "S8e", "_postnatal_legend", variant=True),
        labels=[COCHLEA_DICT[name]["alias"] for name in PAPER_COHORTS["fchrimson_gerbil_postnatal"]])


@panel("S9e")
def supp_fig9e(args):
    lavision_data = _load("lavision", lambda: data.get_lavision_otof_data(
        _cache(args, "otof_data_LaVision.pkl"), args.refresh_cache))
    plot_cochlea_legend(out(args, "S9e", "_legend"), data.LAVISION_COCHLEAE_DICT, ncol=2)
    gene_therapy.plot_lavision_efficiency_per_band(
        lavision_data, out(args, "S9e"), plot=args.plot, use_alias=args.use_alias, trendline_mode="mean",
        mapping="mueller")


def select_panels(tokens):
    """Get the panel ids for the command line tokens, in figure order.

    A token is a panel id like "2c" or "S4b", or a figure like "2" or "S4" for all of its panels.
    """
    if not tokens:
        return list(PANELS)
    selected = set()
    for token in tokens:
        token = token[:1].upper() + token[1:].lower() if token[:1] in "sS" else token.lower()
        matches = [panel_id for panel_id in PANELS if re.fullmatch(rf"{re.escape(token)}[a-z]?", panel_id)]
        if not matches:
            raise ValueError(f"Unknown panel or figure '{token}'. Available panels: {', '.join(PANELS)}.")
        selected.update(matches)
    return [panel_id for panel_id in PANELS if panel_id in selected]


def main():
    parser = argparse.ArgumentParser(description="Create the plot panels of the CochleaNet paper.")
    parser.add_argument("output_dir", help="Output directory for the panels.")
    parser.add_argument("-p", "--panels", nargs="+", default=None,
                        help="Panels (e.g. 2c S4b) or figures (e.g. 2 S4) to create. Default: all.")
    parser.add_argument("--variants", action="store_true",
                        help="Also create the unpublished variants of the selected panels in <output_dir>/variants.")
    parser.add_argument("--napari", action="store_true", help="Show the napari renderings of panels 1a and 3a.")
    parser.add_argument("--plot", action="store_true", help="Show every figure after saving it.")
    parser.add_argument("--no_alias", action="store_true", help="Label cochleae by name instead of alias.")
    parser.add_argument("--refresh_cache", action="store_true",
                        help="Ignore the cached pickles and fetch the data again.")
    parser.add_argument("--cache_dir", default=SCRIPT_DIR,
                        help=f"Directory of the pickle caches. Default: {SCRIPT_DIR}.")
    parser.add_argument("--data_dir", default=os.path.join(REPO_ROOT, "reproducibility", "model_accuracy"),
                        help="Directory with the model and annotator accuracy JSON files.")
    parser.add_argument("--synapse_dir", default=SYNAPSE_DIR_ROOT,
                        help="Directory with the ihc_counts_<version> folders of synapse count tables.")
    parser.add_argument("--variance_dir", default=None,
                        help="Directory with the *_variance.json files of the annotators, for panel S6d.")
    args = parser.parse_args()
    args.use_alias = not args.no_alias

    try:
        panel_ids = select_panels(args.panels)
    except ValueError as e:
        parser.error(str(e))

    failed = []
    done = set()
    for panel_id in panel_ids:
        functions = [PANELS[panel_id]] + ([VARIANTS[panel_id]] if args.variants and panel_id in VARIANTS else [])
        for function in functions:
            if function in done:
                continue
            done.add(function)
            # Every panel starts from the matplotlib defaults, like a run of a single figure script.
            matplotlib.rc_file_defaults()
            plt.close("all")
            print(f"Panel {panel_id}: {function.__name__}")
            try:
                function(args)
            except Exception:
                traceback.print_exc()
                failed.append(f"{panel_id} ({function.__name__})")

    if failed:
        print(f"{len(failed)} panel function(s) failed: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
