import sys

import matplotlib.pyplot as plt

sys.path.append("../../figures")


def plot_tonotopy(bin_labels, values, names, expression_eff_list):
    from util import prism_style, prism_cleanup_axes

    color_dict = {"O1": "#9C5027", "O2": "#67279C"}

    prism_style()
    label_size = 20
    tick_label_size = 14

    # result = pd.DataFrame(result)
    # bin_labels = pd.unique(result["octave_band"])
    # band_to_x = {band: i for i, band in enumerate(bin_labels)}
    # result["x_pos"] = result["octave_band"].map(band_to_x)

    fig, ax = plt.subplots(figsize=(8, 4))

    offset = 0.08
    for num in range(len(names)):

        name = names[num]
        expr_vals = values[num]

        x_positions = range(len(bin_labels))
        x_positions = [x - len(bin_labels) // 2 * offset + offset * num for x in x_positions]

        x_positions = [pos for i, pos in enumerate(x_positions) if expr_vals[i] is not None]
        expr_vals = [val for val in expr_vals if val is not None]
        assert len(x_positions) == len(expr_vals)

        ax.scatter(x_positions, expr_vals, marker="o", label=name, s=80, alpha=1, color=color_dict[name])

    xlim_left, xlim_right = ax.get_xlim()
    y_offset = [0.01, -0.04]
    x_offset = 0.5
    plt.xlim(xlim_left, xlim_right)
    for num, key in enumerate(color_dict.keys()):
        color = color_dict[key]
        expression_eff = expression_eff_list[num]

        ax.text(xlim_left + x_offset, expression_eff + y_offset[num], "mean",
                color=color, fontsize=tick_label_size, ha="center")
        trend_r, = ax.plot(
            [xlim_left, xlim_right],
            [expression_eff, expression_eff],
            linestyle="dashed",
            color=color,
            alpha=0.7,
            zorder=0
        )

    ax.set_xticks(range(len(bin_labels)))
    ax.set_xticklabels(bin_labels)
    ax.set_xlabel("Octave band [kHz]", fontsize=label_size)

    ax.set_ylabel("Expression efficiency")
    plt.tight_layout()
    prism_cleanup_axes(ax)

    plt.show()


def main():
    frequencies = ["4-8", "8-12", "12-16", "16-24", "24-32"]

    n_ihcs_23R = [51, 107, 77, 105, None]
    n_pos_23R = [15, 36, 24, 29, None]
    assert len(n_ihcs_23R) == len(n_pos_23R)
    assert len(n_ihcs_23R) == len(frequencies)

    n_ihcs_25R = [None, 103, 71, 102, 70]
    n_pos_25R = [None, 24, 21, 28, 2]
    assert len(n_ihcs_25R) == len(n_pos_25R)
    assert len(n_ihcs_25R) == len(frequencies)

    total_23R = sum(n for n in n_ihcs_23R if n is not None)
    pos_23R = sum(n for n in n_pos_23R if n is not None)
    eff_23R = float(pos_23R) / total_23R
    print("N-IHCs 23R:", total_23R)
    print("Expr. Eff.:", eff_23R)
    print()

    total_25R = sum(n for n in n_ihcs_25R if n is not None)
    pos_25R = sum(n for n in n_pos_25R if n is not None)
    eff_25R = float(pos_25R) / total_25R
    print("N-IHCs 25R:", total_25R)
    print("Expr. Eff.:", eff_25R)

    expr_23R = [None if n is None else float(pos) / n
                for pos, n in zip(n_pos_23R, n_ihcs_23R)]
    expr_25R = [None if n is None else float(pos) / n
                for pos, n in zip(n_pos_25R, n_ihcs_25R)]
    plot_tonotopy(
        frequencies,
        values=[expr_23R, expr_25R],
        names=["O1", "O2"],
        expression_eff_list=[eff_23R, eff_25R]
    )


if __name__ == "__main__":
    main()
