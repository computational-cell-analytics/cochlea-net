import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from flamingo_tools.s3_utils import create_s3_target, BUCKET_NAME

sys.path.append("../figures")


def _gaussian_kernel1d(sigma_pix, truncate=4.0):
    r = int(truncate * sigma_pix + 0.5)
    x = np.arange(-r, r + 1, dtype=float)
    k = np.exp(-(x**2) / (2 * sigma_pix**2))
    k /= k.sum()
    return k


def density_1d_simple(x, window):
    x = np.asarray(x, dtype=float)
    xmin, xmax = x.min(), x.max()

    # Fixed grid (kept internal for simplicity)
    nbins = 256
    edges = np.linspace(xmin, xmax, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    dx = edges[1] - edges[0]

    # Bin counts, smooth with Gaussian, convert to density
    counts, _ = np.histogram(x, bins=edges, density=False)
    sigma_pix = max(window / dx, 1e-9)
    k = _gaussian_kernel1d(sigma_pix)
    counts_smooth = np.convolve(counts, k, mode="same")
    density = counts_smooth / dx

    return centers, density


def open_tsv(fs, path):
    s3_path = os.path.join(BUCKET_NAME, path)
    with fs.open(s3_path, "r") as f:
        table = pd.read_csv(f, sep="\t")
    return table


def analyze_cochlea(cochlea, plot=False):
    fs = create_s3_target()
    seg_name = "SGN_v2"

    table_path = f"tables/{seg_name}/default.tsv"
    table = open_tsv(fs, os.path.join(cochlea, table_path))

    component_ids = [1]
    table = table[table.component_labels.isin(component_ids)]

    window_size = 200.0
    grid, density = density_1d_simple(table["length[µm]"], window=window_size)  # window in same units as x

    if len(grid) != len(density):
        breakpoint()

    if plot:
        plt.figure(figsize=(6, 3))
        plt.plot(grid, density, lw=2)
        plt.xlabel("Length [µm]")
        plt.ylabel("Density [SGN/µm]")
        plt.tight_layout()
        plt.show()
    else:
        return grid, density


def get_sgn_counts(cochlea):
    fs = create_s3_target()
    seg_name = "SGN_v2"

    table_path = f"tables/{seg_name}/default.tsv"
    table = open_tsv(fs, os.path.join(cochlea, table_path))
    component_ids = [1]
    table = table[table.component_labels.isin(component_ids)]

    frequencies = table["frequency[kHz]"].values
    values = np.ones_like(frequencies)

    return frequencies, values


def average_densities(curves, *, nbins=512, weights=None, renormalize=True):
    if len(curves) == 0:
        raise ValueError("curves must be non-empty")

    # Global domain across all inputs
    xmin = min(g[0][0] for g in curves)
    xmax = max(g[0][-1] for g in curves)
    if not np.isfinite([xmin, xmax]).all() or xmax <= xmin:
        raise ValueError("Invalid global domain from inputs.")

    grid_common = np.linspace(xmin, xmax, nbins)
    interp_dens = []

    for grid, dens in curves:
        grid = np.asarray(grid, float)
        dens = np.asarray(dens, float)
        # Interpolate onto common grid; outside each curve's support -> 0
        interp = np.interp(grid_common, grid, dens, left=0.0, right=0.0)
        # Clip tiny negatives that may appear from numeric noise
        interp_dens.append(np.clip(interp, 0.0, np.inf))

    M = np.vstack(interp_dens)  # shape: (n_curves, nbins)

    if weights is None:
        w = np.ones(M.shape[0], float)
    else:
        w = np.asarray(weights, float)
        if w.shape[0] != M.shape[0]:
            raise ValueError("weights must have same length as number of curves")
        if np.any(w < 0):
            raise ValueError("weights must be non-negative")
    w = w / w.sum()

    mean_density = (w[:, None] * M).sum(axis=0)

    if renormalize:
        area = np.trapz(mean_density, grid_common)
        if area > 0:
            mean_density /= area

    return grid_common, mean_density


def check_implementation():
    cochlea = "G_EK_000049_L"
    analyze_cochlea(cochlea, plot=True)


def compare_cochleae(cochleae, animal, plot_density=True, plot_tonotopy=True, alias=None):
    if alias is not None:
        assert len(alias) == len(cochleae)

    if plot_density:
        plt.figure(figsize=(6, 3))
        for i, cochlea in enumerate(cochleae):
            grid, density = analyze_cochlea(cochlea, plot=False)
            plt.plot(grid, density, lw=2, label=cochlea if alias is None else alias[i])

        plt.xlabel("Length [µm]")
        plt.ylabel("Density [SGN/µm]")
        plt.legend()
        plt.tight_layout()
        plt.show()

    if plot_tonotopy:
        from util import frequency_mapping

        fig, ax = plt.subplots(figsize=(6, 3))
        for cochlea in cochleae:
            frequencies, values = get_sgn_counts(cochlea)
            sgns_per_band = frequency_mapping(
                frequencies, values, animal=animal, aggregation="sum"
            )
            bin_labels = sgns_per_band.index
            binned_counts = sgns_per_band.values

            band_to_x = {band: i for i, band in enumerate(bin_labels)}
            x_positions = bin_labels.map(band_to_x)
            ax.scatter(x_positions, binned_counts, marker="o", s=80, label=cochlea if alias is None else alias[i])

        ax.set_xticks(range(len(bin_labels)))
        ax.set_xticklabels(bin_labels)
        ax.set_xlabel("Octave band [kHz]")
        ax.legend()
        plt.show()


def compare_cochlea_groups(cochlea_groups, animal, plot_density=True, plot_tonotopy=True):

    if plot_density:
        fix, axes = plt.subplots(2, sharey=True, sharex=True)
        for name, cochleae in cochlea_groups.items():
            group_values = []
            for cochlea in cochleae:
                grid, density = analyze_cochlea(cochlea, plot=False)
                axes[0].plot(grid, density, lw=1, label=cochlea, alpha=0.8)
                group_values.append((grid, density))
            group_grid, group_density = average_densities(group_values, nbins=len(grid), renormalize=False)
            axes[1].plot(group_grid, group_density, label=name, lw=2)

        for ax in axes:
            ax.set_xlabel("Length [µm]")
            ax.set_ylabel("Density [SGN/µm]")
            ax.legend()
        plt.tight_layout()
        plt.show()

    if plot_tonotopy:
        from util import frequency_mapping

        fig, axes = plt.subplots(2, sharey=True)
        for name, cochleae in cochlea_groups.items():
            grp_values = []
            for cochlea in cochleae:
                frequencies, values = get_sgn_counts(cochlea)
                sgns_per_band = frequency_mapping(
                    frequencies, values, animal=animal, aggregation="sum"
                )
                bin_labels = sgns_per_band.index
                binned_counts = sgns_per_band.values

                band_to_x = {band: i for i, band in enumerate(bin_labels)}
                x_positions = bin_labels.map(band_to_x)
                axes[0].scatter(x_positions, binned_counts, marker="o", label=cochlea, s=80)

                grp_values.append(binned_counts)

            grp_values = np.array(grp_values)
            grp_mean = grp_values.mean(axis=0)
            grp_std = grp_values.std(axis=0)

            axes[1].plot(x_positions, grp_mean, lw=2, label=name)
            axes[1].fill_between(x_positions, grp_mean - grp_std, grp_mean + grp_std, alpha=0.3)

        for ax in axes:
            ax.set_xticks(range(len(bin_labels)))
            ax.set_xticklabels(bin_labels)
            ax.set_xlabel("Octave band [kHz]")
            ax.set_ylabel("SGN Count")
            ax.legend()
        plt.show()


def density_analysis_gerbil():
    # Comparison for Gerbil.
    cochleae = ["G_EK_000233_L", "G_EK_000049_L", "G_EK_000049_R"]
    alias = ["G_1L", "G_2L", "G_2R"]
    compare_cochleae(cochleae, animal="gerbil", plot_density=True, alias=alias)


def density_analysis_mice():
    # Comparison for Mouse.
    # This is for Aleyna's SFN talk.

    # NOTE: There is some problem with M_LR_000143_L and "M_LR_000153_L"
    # I have removed the corresponding pairs for now, but we should investigate and add back.
    # Healthy reference cochleae.
    reference_cochleae = [
        "M_LR_000226_L", "M_LR_000226_R", "M_LR_000227_L", "M_LR_000227_R",
    ]
    # Right un-injected cochleae.
    uninjected_cochleae = [
        "M_LR_000144_R", "M_LR_000145_R",  "M_LR_000155_R", "M_LR_000189_R",
    ]
    # Left injected cochleae.
    injected_cochleae = [
        "M_LR_000144_L", "M_LR_000145_L", "M_LR_000155_L", "M_LR_000189_L",
    ]
    compare_cochlea_groups(
        {
            "reference": reference_cochleae,
            "uninjected": uninjected_cochleae,
            "injected": injected_cochleae,
        },
        animal="mouse", plot_tonotopy=True, plot_density=True,
    )


# The visualization has to be improved to make plots understandable.
def main():
    # check_implementation()
    density_analysis_gerbil()
    # density_analysis_mice()


if __name__ == "__main__":
    main()
