import json
import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.filters import threshold_otsu

from flamingo_tools.s3_utils import BUCKET_NAME, create_s3_target
from flamingo_tools.measurements import compute_object_measures
from flamingo_tools.segmentation.sgn_subtype_utils import STAIN_TO_TYPE, COCHLEAE


# Define the animal specific octave bands.
def _get_mapping(animal):
    if animal == "mouse":
        bin_edges = [0, 2, 4, 8, 16, 32, 64, np.inf]
        bin_labels = [
            "<2", "2–4", "4–8", "8–16", "16–32", "32–64", ">64"
        ]
    elif animal == "gerbil":
        bin_edges = [0, 0.5, 1, 2, 4, 8, 16, 32, np.inf]
        bin_labels = [
            "<0.5", "0.5–1", "1–2", "2–4", "4–8", "8–16", "16–32", ">32"
        ]
    else:
        raise ValueError
    assert len(bin_edges) == len(bin_labels) + 1
    return bin_edges, bin_labels


def frequency_mapping(frequencies, values, animal="mouse", transduction_efficiency=False):
    # Get the mapping of frequencies to octave bands for the given species.
    bin_edges, bin_labels = _get_mapping(animal)

    # Construct the data frame with octave bands.
    df = pd.DataFrame({"freq_khz": frequencies, "value": values})
    df["octave_band"] = pd.cut(
        df["freq_khz"], bins=bin_edges, labels=bin_labels, right=False
    )

    if transduction_efficiency:  # We compute the transduction efficiency per band.
        num_pos = df[df["value"] == 1].groupby("octave_band", observed=False).size()
        num_tot = df[df["value"].isin([1, 2])].groupby("octave_band", observed=False).size()
        value_by_band = (num_pos / num_tot).reindex(bin_labels)
    else:  # Otherwise, aggregate the values over the octave band using the mean.
        value_by_band = (
            df.groupby("octave_band", observed=True)["value"]
              .sum()
              .reindex(bin_labels)   # keep octave order even if a bin is empty
        )
    return value_by_band


# Map from cochlea names to channels
COCHLEAE_FOR_SUBTYPES = {
    "M_LR_000098_L": ["PV", "CR", "Ntng1"],
    "M_LR_000099_L": ["PV", "Calb1", "Lypd1"],
    "M_LR_000184_R": ["PV", "Prph"],
    "M_LR_000184_L": ["PV", "Prph"],
    "M_LR_000260_L": ["PV", "Prph", "Tuj1"],
    "M_LR_N110_L": ["PV", "Calb1", "Ntng1"],
    "M_LR_N110_R": ["PV", "Calb1", "Ntng1"],
    "M_LR_N152_L": ["PV", "CR", "Ntng1"],
    "M_AMD_N180_L": ["CR", "Ntng1"],
    "M_AMD_N180_R": ["CR", "Ntng1"],
    "M_AMD_Runx1_L": ["PV", "CR", "Ntng1"],
}

GROUPINGS = {
    "Type Ia;Type Ib;Type Ic": ["M_LR_000098_L", "M_LR_N152_L", "M_AMD_N180_L", "M_AMD_N180_R"],
    "Type I;Type II": ["M_LR_000184_L", "M_LR_000184_R", "M_LR_000260_L"],
    "Type Ib;Type Ic": ["M_LR_N110_L", "M_LR_N110_R"],
    "Type Ib;Type Ic;Type IbIc": ["M_LR_000099_L"],
}


REGULAR_COCHLEAE = [
    "M_LR_000098_L", "M_LR_000099_L", "M_LR_000184_R", "M_LR_000184_L", "M_LR_000260_L",
    "M_LR_N110_L", "M_LR_N110_R", "M_LR_N152_L",
    "M_AMD_N180_L", "M_AMD_N180_R",
]

# For consistent colors.
ALL_COLORS = ["red", "blue", "orange", "yellow", "cyan", "magenta", "green", "purple", "gray", "black"]
COLORS = {
    "Type Ia": "#133374",
    "Type Ib": "#27339C",
    "Type IbIc": "#67279C",
    "Type Ic": "#9C276F",
    "inconclusive": "#9C8227",

    "Type I": "#9C3B27",
    "Type II": "#279C96",
    "default": "#279C47"
}

PLOT_OUT = "./subtype_plots"


# Type Ia ; CR+ / Calb1- or Calb1- / Lypd1-
# Type Ib: CR+ / Calb1+ or Calb1+ / Lypd1+
# Type Ic: CR-/Calb1+ - or Calb1- / Lypd1+
# Type II: CR-/Calb1- or Calb1- / Lypd1- or Prph+
def stain_to_type(stain):
    # Normalize the staining string.
    stains = stain.replace(" ", "").split("/")
    assert len(stains) in (1, 2)

    if len(stains) == 1:
        stain_norm = stain
    else:
        s1, s2 = sorted(stains)
        stain_norm = f"{s1}/{s2}"

    if stain_norm not in STAIN_TO_TYPE:
        print(stain_norm)
        breakpoint()
        raise ValueError(f"Invalid stain combination: {stain_norm}")

    return STAIN_TO_TYPE[stain_norm], stain_norm


def check_processing_status():
    s3 = create_s3_target()

    # For checking the dataset names.
    # content = s3.open(f"{BUCKET_NAME}/project.json", mode="r", encoding="utf-8")
    # info = json.loads(content.read())
    # datasets = info["datasets"]
    # for name in datasets:
    #     print(name)
    # breakpoint()

    missing_tables = {}

    for cochlea, channels in COCHLEAE_FOR_SUBTYPES.items():
        if cochlea not in REGULAR_COCHLEAE:
            continue
        try:
            content = s3.open(f"{BUCKET_NAME}/{cochlea}/dataset.json", mode="r", encoding="utf-8")
        except FileNotFoundError:
            print(cochlea, "is not yet on MoBIE")
            print()
            continue
        info = json.loads(content.read())
        sources = info["sources"]

        channels_found = [name for name in channels if name in sources]
        channels_missing = [name for name in channels if name not in sources]

        print("Cochlea:", cochlea)
        print("Found the expected channels:", channels_found)
        if channels_missing:
            print("Missing the expected channels:", channels_missing)

        if "SGN_v2b" in sources:
            print("SGN segmentation is present with name SGN_v2b")
            table_folder = "tables/SGN_v2b"
        elif "SGN_v2" in sources:
            print("SGN segmentation is present with name SGN_v2")
            table_folder = "tables/SGN_v2"
        elif "PV_SGN_v2" in sources:
            print("SGN segmentation is present with name PV_SGN_v2")
            table_folder = "tables/PV_SGN_v2"
        elif "CR_SGN_v2" in sources:
            print("SGN segmentation is present with name CR_SGN_v2")
            table_folder = "tables/CR_SGN_v2"
        else:
            print("SGN segmentation is MISSING")
            print()
            continue

        # Check which tables we have.
        expected_tables = ["default.tsv"]

        tables = s3.ls(os.path.join(BUCKET_NAME, cochlea, table_folder))
        tables = [os.path.basename(tab) for tab in tables]

        this_missing_tables = []
        for exp_tab in expected_tables:
            if exp_tab not in tables:
                print("Missing table:", exp_tab)
                this_missing_tables.append(exp_tab)
        missing_tables[cochlea] = this_missing_tables
        print()

    return missing_tables


def require_missing_tables(missing_tables):
    output_root = "./object_measurements"

    for cochlea, missing_tabs in missing_tables.items():
        if cochlea not in REGULAR_COCHLEAE:
            continue
        for missing in missing_tabs:
            channel = missing.split("_")[0]
            seg_name = missing.split("_")[1].replace("-", "_")
            print("Computing intensities for cochlea:", cochlea, "segmentation:", seg_name, "channel:", channel)

            img_s3 = f"{cochlea}/images/ome-zarr/{channel}.ome.zarr"
            seg_s3 = f"{cochlea}/images/ome-zarr/{seg_name}.ome.zarr"
            seg_table_s3 = f"{cochlea}/tables/{seg_name}/default.tsv"
            # img_path, _ = get_s3_path(img_s3)
            # seg_path, _ = get_s3_path(seg_s3)

            output_folder = os.path.join(output_root, cochlea)
            os.makedirs(output_folder, exist_ok=True)
            output_table_path = os.path.join(
                output_folder, f"{channel}_{seg_name.replace('_', '-')}_object-measures.tsv"
            )
            compute_object_measures(
                image_path=img_s3,
                segmentation_path=seg_s3,
                segmentation_table_path=seg_table_s3,
                output_table_path=output_table_path,
                image_key="s0",
                segmentation_key="s0",
                s3_flag=True,
                component_list=[1],
                n_threads=16,
            )

            # S3 upload
            # from subprocess import run
            # run(["rclone", "--progress", "copyto", output_folder,
            #      f"cochlea-lightsheet:cochlea-lightsheet/{cochlea}/tables/{seg_name}"])


def compile_data_for_subtype_analysis():
    s3 = create_s3_target()

    output_folder = "./subtype_analysis"
    os.makedirs(output_folder, exist_ok=True)

    for cochlea, channels in COCHLEAE_FOR_SUBTYPES.items():
        if cochlea not in REGULAR_COCHLEAE:
            continue
        if "PV" in channels:
            reference_channel = "PV"
            seg_name = "PV_SGN_v2"
        else:
            assert "CR" in channels
            reference_channel = "CR"
            seg_name = "CR_SGN_v2"
        print(cochlea)

        content = s3.open(f"{BUCKET_NAME}/{cochlea}/dataset.json", mode="r", encoding="utf-8")
        info = json.loads(content.read())
        sources = info["sources"]

        # Load the segmentation table.
        try:
            seg_source = sources[seg_name]
        except KeyError as e:
            if seg_name == "PV_SGN_v2":
                if "output_seg" in list(COCHLEAE[cochlea].keys()):
                    seg_source = sources[COCHLEAE[cochlea]["output_seg"]]
                    seg_name = COCHLEAE[cochlea]["output_seg"]
                else:
                    seg_source = sources[COCHLEAE[cochlea]["seg_data"]]
                    seg_name = COCHLEAE[cochlea]["seg_data"]
            else:
                raise e
        table_folder = os.path.join(
            BUCKET_NAME, cochlea, seg_source["segmentation"]["tableData"]["tsv"]["relativePath"]
        )
        table_content = s3.open(os.path.join(table_folder, "default.tsv"), mode="rb")
        table = pd.read_csv(table_content, sep="\t")

        # Get the SGNs in the main component
        component_list = [1]
        if "component_list" in list(COCHLEAE[cochlea].keys()):
            component_list = COCHLEAE[cochlea]["component_list"]
        table = table[table["component_labels"].isin(component_list)]
        print("Number of SGNs", len(table))
        valid_sgns = table.label_id

        output_table = {"label_id": table.label_id.values, "frequency[kHz]": table["frequency[kHz]"]}

        # Analyze the different channels (= different subtypes).
        reference_intensity = None
        for channel in channels:
            # Load the intensity table, prefer local.
            table_folder = os.path.join(
                BUCKET_NAME, cochlea, seg_source["segmentation"]["tableData"]["tsv"]["relativePath"]
            )
            table_content = s3.open(os.path.join(table_folder, "default.tsv"), mode="rb")
            table = pd.read_csv(table_content, sep="\t")
            table = table[table["component_labels"].isin(component_list)]

            # local
            table_name = f"{channel}_{seg_name.replace('_', '-')}_object-measures.tsv"
            intensity_path = os.path.join("object_measurements", cochlea, table_name)

            if os.path.exists(intensity_path):
                intensities = pd.read_csv(intensity_path, sep="\t")
            else:
                intensity_path = os.path.join(table_folder, table_name)
                table_content = s3.open(intensity_path, mode="rb")

                intensities = pd.read_csv(table_content, sep="\t")
                intensities = intensities[intensities.label_id.isin(valid_sgns)]
            assert len(table) == len(intensities)
            assert (intensities.label_id.values == table.label_id.values).all()

            medians = intensities["median"].values
            output_table[f"{channel}_median"] = medians
            if channel == reference_channel:
                reference_intensity = medians
            else:
                assert reference_intensity is not None
                output_table[f"{channel}_ratio_{reference_channel}"] = medians / reference_intensity

        out_path = os.path.join(output_folder, f"{cochlea}_subtype_analysis.tsv")
        output_table = pd.DataFrame(output_table)
        output_table.to_csv(out_path, sep="\t", index=False)


def _plot_histogram(table, column, name, show_plots, class_names=None, apply_threshold=True):
    data = table[column].values
    threshold = threshold_otsu(data)

    if class_names is not None:
        assert len(class_names) == 2
        c0, c1 = class_names
        subtype_classification = [c0 if datum < threshold else c1 for datum in data]

    fig, ax = plt.subplots(1)
    ax.hist(data, bins=24)
    if apply_threshold:
        ax.axvline(x=threshold, color='red', linestyle='--')
        if class_names is None:
            ax.set_title(f"{name}\n threshold: {threshold}")
        else:
            pos_perc = len([st for st in subtype_classification if st == c1]) / float(len(subtype_classification))
            ax.set_title(f"{name}\n threshold: {threshold}\n %{c1}: {pos_perc * 100}")
    else:
        ax.set_title(name)

    if show_plots:
        plt.show()
    else:
        os.makedirs(PLOT_OUT, exist_ok=True)
        plt.savefig(f"{PLOT_OUT}/{name}.png")
    plt.close()

    if class_names is not None:
        return subtype_classification


def _plot_2d(ratios, name, show_plots, classification=None, colors=None):
    fig, ax = plt.subplots(1)
    assert len(ratios) == 2
    keys = list(ratios.keys())
    k1, k2 = keys

    if classification is None:
        ax.scatter(ratios[k1, k2])

    else:
        assert colors is not None
        unique_labels = set(classification)
        for lbl in unique_labels:
            mask = [ll == lbl for ll in classification]
            ax.scatter(
                [ratios[k1][i] for i in range(len(classification)) if mask[i]],
                [ratios[k2][i] for i in range(len(classification)) if mask[i]],
                c=colors[lbl], label=lbl
            )

        ax.legend()

    ax.set_xlabel(k1)
    ax.set_ylabel(k2)
    ax.set_title(name)

    if show_plots:
        plt.show()
    else:
        os.makedirs(PLOT_OUT, exist_ok=True)
        plt.savefig(f"{PLOT_OUT}/{name}.png")
    plt.close()


def _plot_tonotopic_mapping(freq, classification, name, colors, show_plots):

    frequency_mapped = frequency_mapping(freq, classification)
    result = next(iter(frequency_mapped.values()))
    bin_labels = pd.unique(result["octave_band"])
    band_to_x = {band: i for i, band in enumerate(bin_labels)}
    x_positions = result["octave_band"].map(band_to_x)

    fig, ax = plt.subplots(figsize=(8, 4))
    for cat, vals in frequency_mapped.items():
        ax.scatter(x_positions, vals.value, label=cat, color=colors[cat])

    main_ticks = range(len(bin_labels))
    ax.set_xticks(main_ticks)
    ax.set_xticklabels(bin_labels)
    ax.set_xlabel("Octave band (kHz)")
    ax.legend()
    ax.set_title(name)

    if show_plots:
        plt.show()
    else:
        os.makedirs(PLOT_OUT, exist_ok=True)
        plt.savefig(f"{PLOT_OUT}/{name}.png")
    plt.close()

    return frequency_mapped


# Combined visualization for the cochleae
# Can we visualize the tonotopy in subtypes and not stainings?
# It would also be good to have subtype percentages per cochlea and pooled together as a diagram and tonotopy?
# This would help to see if different staining gives same/similar results.
def combined_analysis(results, show_plots):
    #
    # Create the tonotopic mapping.
    #
    summary = {}
    colors = {}
    for cochlea, result in results.items():
        if cochlea == "M_LR_000214_L":  # One of the signals cannot be analyzed.
            continue
        classification = result["classification"]
        frequencies = result["frequencies"]
        # get categories
        cats = list(set([c[:c.find(" (")] for c in classification]))
        cats.sort()

        dic = {}
        for c in cats:
            sub_freq = [frequencies[i] for i in range(len(classification))
                        if classification[i][:classification[i].find(" (")] == c]
            mapping = frequency_mapping(sub_freq, [1 for _ in range(len(sub_freq))])
            mapping.fillna(0, inplace=True)

            mapping = mapping.astype('float32')
            dic[c] = mapping
            bin_labels = pd.unique(mapping.index)

            if c not in colors:
                current_colors = list(colors.values())
                next_color = ALL_COLORS[len(current_colors)]
                colors[c] = next_color

        for bin in bin_labels:
            total = sum([dic[key][bin] for key in dic.keys()])
            for key in dic.keys():
                dic[key][bin] = float(dic[key][bin] / total)

        summary[cochlea] = dic

    for group, cochleae in GROUPINGS.items():
        sub_dic = {c: summary[c] for c in cochleae if c in summary}

        group_str = "-".join(("".join(group.split(" "))).split(";"))
        save_path = f"./subtype_plots/tonotopic_mapping_{group_str}.png"

        fig, axes = plt.subplots(len(sub_dic), sharey=True, figsize=(8, 8))
        for i, (cochlea, dic) in enumerate(sub_dic.items()):
            types = list(dic.keys())
            if len(sub_dic) == 1:
                ax = axes
            else:
                ax = axes[i]
            for cat in types:
                frequency_mapped = dic[cat]
                bin_labels = pd.unique(frequency_mapped.index)
                x_positions = [i for i in range(len(bin_labels))]
                values = frequency_mapped.values
                if cat in COLORS.keys():
                    color = COLORS[cat]
                else:
                    color = COLORS["default"]
                sc = ax.scatter(x_positions, values, label=cat, color=color)

            main_ticks = range(len(bin_labels))
            ax.set_xticks(main_ticks)
            ax.set_xticklabels(bin_labels)
            ax.set_title(cochlea)
            ax.set_ylabel("Fraction")
            handles, labels = ax.get_legend_handles_labels()

        ax.set_xlabel("Octave band (kHz)")
        fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.85, 0.5), title='Subtypes') #1.02

        plt.tight_layout(rect=[0, 0, 0.85, 1])  # make space for legend
        if show_plots:
            plt.show()
        else:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()

    #
    # Create the overview figure.
    #

    for group, cochleae in GROUPINGS.items():
        group_str = "-".join(("".join(group.split(" "))).split(";"))
        save_path = f"./subtype_plots/{group_str}.png"

        summary, types = {}, []
        for cochlea, result in results.items():
            if cochlea not in cochleae:  # One of the signals cannot be analyzed.
                continue

            classification = result["classification"]
            classification = [cls[:cls.find(" (")] for cls in classification]
            n_tot = len(classification)

            this_types = list(set(classification))
            types.extend(this_types)
            summary[cochlea] = {}
            for stype in types:
                n_type = len([cls for cls in classification if cls == stype])
                type_ratio = float(n_type) / n_tot
                summary[cochlea][stype] = type_ratio

        types = list(set(types))
        types.sort()
        df = pd.DataFrame(summary).fillna(0)  # missing values → 0

        # Transpose → cochleae on x-axis, subtypes stacked
        if len(types) == 6:
            types = [types[2], types[3], types[4], types[5], types[0], types[1]]
        print(types)
        colors = [COLORS[t] for t in types]

        ax = df.T.plot(kind="bar", stacked=True, figsize=(8, 5), color=colors)

        ax.set_ylabel("Fraction")
        ax.set_xlabel("Cochlea")
        ax.set_title("Subtype Fractions per Cochlea")
        plt.legend(loc="lower right")
        plt.xticks(rotation=0)
        plt.tight_layout()

        if show_plots:
            plt.show()
        else:
            plt.savefig(save_path)
            plt.close()


def analyze_subtype_data_regular(show_plots=True):
    global PLOT_OUT, COLORS  # noqa
    PLOT_OUT = "subtype_plots/regular_mice"

    s3 = create_s3_target()

    files = sorted(glob("./subtype_analysis/*.tsv"))
    results = {}

    for ff in files:
        cochlea = os.path.basename(ff)[:-len("_subtype_analysis.tsv")]
        if cochlea not in REGULAR_COCHLEAE:
            continue
        channels = COCHLEAE_FOR_SUBTYPES[cochlea]
        if "PV" in channels:
            reference_channel = "PV"
            seg_name = "PV_SGN_v2"
            stain_channels = channels[1:]
        else:
            assert "CR" in channels
            reference_channel = "CR"
            seg_name = "SGN_merged"
            stain_channels = channels
        assert channels[0] == reference_channel

        content = s3.open(f"{BUCKET_NAME}/{cochlea}/dataset.json", mode="r", encoding="utf-8")
        info = json.loads(content.read())
        sources = info["sources"]

        # Load the segmentation table.
        try:
            seg_source = sources[seg_name]
        except KeyError as e:
            if seg_name == "PV_SGN_v2":
                if "output_seg" in list(COCHLEAE[cochlea].keys()):
                    seg_source = sources[COCHLEAE[cochlea]["output_seg"]]
                    seg_name = COCHLEAE[cochlea]["output_seg"]
                else:
                    seg_source = sources[COCHLEAE[cochlea]["seg_data"]]
                    seg_name = COCHLEAE[cochlea]["seg_data"]
            else:
                raise e
        table_folder = os.path.join(
            BUCKET_NAME, cochlea, seg_source["segmentation"]["tableData"]["tsv"]["relativePath"]
        )
        print("table_folder", table_folder)

        if "component_list" in list(COCHLEAE[cochlea].keys()):
            component_list = COCHLEAE[cochlea]["component_list"]
        else:
            component_list = [1]

        table_content = s3.open(os.path.join(table_folder, "default.tsv"), mode="rb")
        table = pd.read_csv(table_content, sep="\t")
        table = table[table["component_labels"].isin(component_list)]

        print(cochlea)
        print(f"Length of table before filtering: {len(table)}")
        # filter subtype table
        for chan in stain_channels:
            column = f"marker_{chan}"
            table = table.loc[table[column].isin([1, 2])]
            print(f"Length of table after filtering channel {chan}: {len(table)}")

        tab = pd.read_csv(ff, sep="\t")

        # 1.) Plot simple intensity histograms, including otsu threshold.
        for chan in channels:
            column = f"{chan}_median"
            name = f"{cochlea}_{chan}_histogram"
            _plot_histogram(tab, column, name, show_plots, apply_threshold=chan != reference_channel)

        # 2.) Plot ratio histograms, including otsu threshold.
        ratios = {}
        classification = []
        for chan in stain_channels:
            column = f"{chan}_ratio_{reference_channel}"
            # e.g. Calb1_ratio_PV
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
            ratios[f"{chan}_{reference_channel}"] = table[column].values

        # Unify the classification and assign colors
        assert len(classification) in (1, 2)
        if len(classification) == 2:
            cls1, cls2 = classification[0], classification[1]
            assert len(cls1) == len(cls2)
            classification = [f"{c1} / {c2}" for c1, c2 in zip(cls1, cls2)]
            show_2d = True
        else:
            classification = classification[0]
            show_2d = False

        classification = [stain_to_type(cls) for cls in classification]
        classification = [f"{stype} ({stain})" for stype, stain in classification]

        unique_labels = set(classification)
        for label in unique_labels:
            if label in COLORS:
                continue
            else:
                COLORS[label] = ALL_COLORS[0]

        # 3.) Plot tonotopic mapping.
        freq = table["frequency[kHz]"].values
        assert len(freq) == len(classification)
        # tonotopic_mapping = _plot_tonotopic_mapping(
        #    freq, classification, name=name, colors=COLORS, show_plots=show_plots
        # )

        # 4.) Plot 2D space of ratios.
        if show_2d:
            name = f"{cochlea}_2d"
            _plot_2d(ratios, name, show_plots, classification=classification, colors=COLORS)

        results[cochlea] = {"classification": classification, "frequencies": freq}

    combined_analysis(results, show_plots=show_plots)


def export_for_annotation():
    files = sorted(glob("./subtype_analysis/*.tsv"))
    out_folder = "./subtype_analysis/for_mobie_annotation"
    os.makedirs(out_folder, exist_ok=True)

    all_thresholds = {}
    for ff in files:
        cochlea = os.path.basename(ff)[:-len("_subtype_analysis.tsv")]
        if cochlea not in REGULAR_COCHLEAE:
            continue

        channels = COCHLEAE_FOR_SUBTYPES[cochlea]
        if "PV" not in channels:
            continue
        reference_channel = "PV"
        assert channels[0] == reference_channel
        tab = pd.read_csv(ff, sep="\t")

        tab_for_export = {"label_id": tab.label_id.values}
        classification = []
        thresholds = {}

        for chan in channels[1:]:
            data = tab[f"{chan}_ratio_PV"].values
            tab_for_export[f"{chan}_ratio_PV"] = data
            threshold = threshold_otsu(data)
            thresholds[chan] = threshold

            c0, c1 = f"{chan}-", f"{chan}+"
            classification.append([c0 if datum < threshold else c1 for datum in data])

        all_thresholds[cochlea] = thresholds

        if len(classification) == 2:
            cls1, cls2 = classification
            classification = [f"{c1} / {c2}" for c1, c2 in zip(cls1, cls2)]
        else:
            classification = classification[0]
        classification = [stain_to_type(cls) for cls in classification]
        classification = [f"{stype} ({stain})" for stype, stain in classification]

        tab_for_export["classification"] = classification
        tab_for_export = pd.DataFrame(tab_for_export)
        tab_for_export.to_csv(os.path.join(out_folder, f"{cochlea}.tsv"), sep="\t", index=False)

    with open(os.path.join(out_folder, "thresholds.json"), "w") as f:
        json.dump(all_thresholds, f, indent=2, sort_keys=True)


# More TODO:
# > It's good to see that for the N mice the Ntng1C and Lypd1 separate from CR so well on the thresholds.
# Can I visualize these samples ones segmentation masks are done to verify the Ntng1C thresholds?
# As this is a quite clear signal I'm not sure if taking the middle of the histogram would be the best choice.
# The segmentations are in MoBIE already. I need to send you the tables for analyzing the signals. Will send them later.
def main():
    # These scripts are for computing the intensity tables etc.
    missing_tables = check_processing_status()
    print("missing tables",  missing_tables)
    # require_missing_tables(missing_tables)
    compile_data_for_subtype_analysis()

    # This script is for exporting the tables for annotation in MoBIE.
    export_for_annotation()

    # This script is for running the analysis and creating the plots.
    analyze_subtype_data_regular(show_plots=False)

    # TODO
    # analyze_subtype_data_N_mice()

    # CTBP2 stain
    # analyze_subtype_data_syn_mice()


if __name__ == "__main__":
    main()
