import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.ticker import ScalarFormatter, MaxNLocator

base_dir = "../.."

input_file = os.path.join(base_dir, "processed_data", "10x", "sample_iSNVs_per_depth.csv")
final_maf_thresholds_file = os.path.join(base_dir, "processed_data", "10x", "final_maf_thresholds.csv")

figures_dir = "../../figures/supplemental"

DEPTH_SAMPLE_COL = "sample"
ISNV_COL = "iSNV_Count"

N_COLS = 3

plt.style.use("plot_style_settings.mplstyle")


def maf_sort_key(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.inf
    

def positions_to_percent(x, genome_length):
    return (x / genome_length) * 100
    

def read_analysis_maf_lookup(thresholds_file):
    thresholds = pd.read_csv(thresholds_file)

    thresholds["Site"] = thresholds["Site"].astype(str).str.strip()
    thresholds["MAF_analysis"] = pd.to_numeric(
        thresholds["MAF_analysis"],
        errors="coerce"
    )
    return dict(zip(thresholds["Site"], thresholds["MAF_analysis"]))


def plot_site_row(
    fig,
    outer_gs_row,
    site_df,
    row_label,
    analysis_maf,
    genome_length,
    is_top_row=False,
    is_bottom_row=False,
    first_hist_ref=None,
):
    inner_gs = outer_gs_row.subgridspec(
        2,
        N_COLS,
        height_ratios=[0.3, 0.7],
        hspace=0.1,
        wspace=0.5
    )

    if first_hist_ref is None:
        ax_hist = fig.add_subplot(inner_gs[:, 0])
        first_hist_ref = ax_hist
    else:
        ax_hist = fig.add_subplot(inner_gs[:, 0], sharex=first_hist_ref)

    ax_unmasked_top = fig.add_subplot(inner_gs[0, 1])
    ax_unmasked_bottom = fig.add_subplot(inner_gs[1, 1], sharex=ax_unmasked_top)

    ax_masked_top = fig.add_subplot(inner_gs[0, 2])
    ax_masked_bottom = fig.add_subplot(inner_gs[1, 2], sharex=ax_masked_top)


    sample_df = (
        site_df[[DEPTH_SAMPLE_COL, "total_positions_1000x"]]
        .drop_duplicates()
        .copy()
    )

    xvals_all = sample_df["total_positions_1000x"].dropna().to_numpy()
    xvals_all_pct = positions_to_percent(xvals_all, genome_length)

    if len(xvals_all_pct) > 0:
        bin_width = 1
        bins = np.arange(-0.5, 100.5 + bin_width, bin_width)
        weights = np.ones_like(xvals_all_pct, dtype=float) / len(xvals_all_pct)

        ax_hist.hist(
            xvals_all_pct,
            bins=bins,
            weights=weights,
            color="steelblue"
        )
        ax_hist.set_ylim(0, 0.35)
        ax_hist.set_yticks([0.0, 0.1, 0.2, 0.3])
        ax_hist.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))

        unmasked_df = site_df[
            (site_df["Masked"] == False)
            & (site_df["MAF"] == 3)
        ].copy()

        masked_df = site_df[
            (site_df["Masked"] == True)
            & (site_df["MAF"] == analysis_maf)
        ].copy()

        for ax in [ax_unmasked_top, ax_unmasked_bottom]:
            ax.scatter(
                positions_to_percent(unmasked_df["total_positions_1000x"], genome_length),
                unmasked_df[ISNV_COL],
                alpha=0.15,
                s=2.5,
                edgecolor="none"
            )

        for ax in [ax_masked_top, ax_masked_bottom]:
            ax.scatter(
                positions_to_percent(masked_df["total_positions_1000x"], genome_length),
                masked_df[ISNV_COL],
                alpha=0.15,
                s=2.5,
                edgecolor="none"
        )
            
        for ax in [ax_hist, ax_unmasked_top, ax_unmasked_bottom, ax_masked_top, ax_masked_bottom]:
            ax.set_xlim(-2, 102)
            ax.set_xticks(np.arange(0, 101, 20))
        
        unmasked_max = unmasked_df[ISNV_COL].max()
        masked_max = masked_df[ISNV_COL].max()

        if pd.isna(unmasked_max):
            unmasked_max = 200
        if pd.isna(masked_max):
            masked_max = 200
        
        unmasked_top_max = np.ceil((max(200, unmasked_max) * 1.05) / 50) * 50 if unmasked_max > 200 else 210
        masked_top_max = np.ceil((max(200, masked_max) * 1.05) / 50) * 50 if masked_max > 200 else 210

        ax_unmasked_bottom.set_ylim(0, 200)
        ax_unmasked_top.set_ylim(201, unmasked_top_max)
        ax_masked_bottom.set_ylim(0, 200)
        ax_masked_top.set_ylim(201, masked_top_max)

        ax_unmasked_top.spines["bottom"].set_linestyle("--")
        ax_unmasked_top.spines["bottom"].set_color("grey")
        ax_unmasked_top.spines["bottom"].set_linewidth(0.3)
        ax_unmasked_bottom.spines["top"].set_linestyle("--")
        ax_unmasked_bottom.spines["top"].set_color("grey")
        ax_unmasked_bottom.spines["top"].set_linewidth(0.3)

        ax_masked_top.spines["bottom"].set_linestyle("--")
        ax_masked_top.spines["bottom"].set_color("grey")
        ax_masked_top.spines["bottom"].set_linewidth(0.3)
        ax_masked_bottom.spines["top"].set_linestyle("--")
        ax_masked_bottom.spines["top"].set_color("grey")
        ax_masked_bottom.spines["top"].set_linewidth(0.3)

        ax_unmasked_top.tick_params(bottom=False, labelbottom=False)
        ax_masked_top.tick_params(bottom=False, labelbottom=False)

        ax_hist.set_ylabel(f"{row_label}\n\nProportion\nof samples")
        ax_unmasked_bottom.set_ylabel("iSNVs\nper sample")
        ax_masked_bottom.set_ylabel("iSNVs\n per sample")
        ax_unmasked_top.set_ylabel("")
        ax_masked_top.set_ylabel("")

        if is_top_row:
            ax_hist.set_title("Coverage distribution")
            ax_unmasked_top.set_title("No mask, MAF ≥ 3%")
            ax_masked_top.set_title("Masked, analysis MAF")

        if is_bottom_row:
            ax_hist.set_xlabel("% Genome with depth ≥1000x")
            ax_unmasked_bottom.set_xlabel("% Genome with depth ≥1000x")
            ax_masked_bottom.set_xlabel("% Genome with depth ≥1000x")
        else:
            ax_hist.tick_params(axis="x", labelbottom=False)
            ax_unmasked_bottom.tick_params(axis="x", labelbottom=False)
            ax_masked_bottom.tick_params(axis="x", labelbottom=False)
        
        ax_unmasked_top.yaxis.set_major_locator(MaxNLocator(nbins=2, min_n_ticks=2))
        ax_masked_top.yaxis.set_major_locator(MaxNLocator(nbins=2, min_n_ticks=2))

    return first_hist_ref


def main():
    plot_df = pd.read_csv(input_file)
    plot_df["Masked"] = plot_df["Masked"].astype(bool)
    plot_df["MAF"] = pd.to_numeric(plot_df["MAF"], errors="coerce")
    genome_length = int(plot_df["total_positions_1000x"].max())

    analysis_maf_lookup = read_analysis_maf_lookup(final_maf_thresholds_file)

    sites = sorted(plot_df["row_label"].drop_duplicates().tolist())

    n_rows = len(sites)

    fig = plt.figure(figsize=(7, 9))

    outer_gs = fig.add_gridspec(
        n_rows,
        1,
        hspace=0.2
    )

    first_hist_ref = None

    for row_idx, site in enumerate(sites):
        site_df = plot_df[plot_df["row_label"] == site].copy()

        site_name = site_df["Site"].iloc[0]

        if site_name not in analysis_maf_lookup:
            raise ValueError(f"No analysis MAF found for site '{site_name}' in thresholds file")
        
        analysis_maf = analysis_maf_lookup[site_name]

        first_hist_ref = plot_site_row(
            fig=fig,
            outer_gs_row=outer_gs[row_idx],
            site_df=site_df,
            row_label=site,
            analysis_maf=analysis_maf,
            genome_length=genome_length,
            is_top_row=(row_idx == 0),
            is_bottom_row=(row_idx == n_rows - 1),
            first_hist_ref=first_hist_ref,
        )

    fig.text(0.03, 0.90, 'a', fontweight='bold', fontsize=8)
    fig.text(0.35, 0.90, 'b', fontweight='bold', fontsize=8)
    fig.text(0.64, 0.90, 'c', fontweight='bold', fontsize=8)

    plt.tight_layout()
    plot_filename = "figureS5_sample_iSNVs_per_depth.png"
    plot_path = os.path.join(figures_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print("figure saved to:", plot_path)


if __name__ == "__main__":
    main()