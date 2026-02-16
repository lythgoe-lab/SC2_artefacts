import os
import re
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from matplotlib.gridspec import GridSpec


base_processed_dir = "../../processed_data/"
cov_folder = "10x"
figures_dir = "../../figures/supplemental"

depth_thresholds = [10, 100, 1000]
MAF_thresholds = [3, 5, 10, 20]

site_to_folder_map = {
    "NORT ARTIC_3": "NORT_ARTIC_3_ILLUMINA",
    "NORT ARTIC_4": "NORT_ARTIC_4_ILLUMINA",
    "NORT ARTIC_4.1": "NORT_ARTIC_4.1_ILLUMINA",
    "NORW ARTIC_4.1": "NORW_ARTIC_4.1_ILLUMINA",
    "OXON ve-SEQ": "OXON_VeSeq_ILLUMINA",
    "PHEC ARTIC_3": "PHEC_ARTIC_3_ILLUMINA",
    "SANG ARTIC_3": "SANG_ARTIC_3_ILLUMINA",
    "SANG ARTIC_4.1": "SANG_ARTIC_4.1_ILLUMINA"
}
subset_sites = list(site_to_folder_map.keys())

# Build groups
groups = {}
nort = [s for s in subset_sites if s.startswith("NORT")]
sang = [s for s in subset_sites if s.startswith("SANG")]
groups['NORT'] = nort
groups['SANG'] = sang
for site in subset_sites:
    if site not in nort + sang:
        key = site.split()[0]
        groups[key] = [site]
group_order = ["SANG", "NORW", "NORT", "PHEC", "OXON"]


def load_sample_totals(mapping):
    fn = os.path.join(base_processed_dir, 'samples_per_site_cov_summary.csv')
    df = pd.read_csv(fn)
    df = df[df['Cov_Threshold'] == cov_folder]
    totals = {}
    for grp, sites in mapping.items():
        total = 0
        for s in sites:
            site_key = site_to_folder_map[s]
            row = df[df['Site'] == site_key]
            if row.empty:
                continue
            total += int(row.iloc[0]['Total_Samples'])
        totals[grp] = total
    return totals


def fmt_reads(x):
    """Formatting for min read count"""
    if x < 1:
        return f"{x:.2f}"
    else:
        return f"{x:.0f}"


pattern = re.compile(r"Samples_MAF_(?P<MAF>\d+)%_depth_(?P<depth>\d+)")

sample_totals = load_sample_totals(groups)


scatter_dfs = []
for grp, sites in groups.items():
    tot = sample_totals.get(grp, 0)
    if tot == 0:
        continue

    dfs = []
    for s in sites:
        path = os.path.join(base_processed_dir, cov_folder, site_to_folder_map[s], 'samples_per_iSNV_position.csv')
        if not os.path.exists(path):
            continue

        df = pd.read_csv(path)
        long = df.melt(id_vars=['Position'], var_name='Threshold', value_name='Num_Samples')
        parsed = long['Threshold'].str.extract(pattern).astype(int)
        long['MAF']   = parsed['MAF']
        long['depth'] = parsed['depth']
        dfs.append(long)

    if not dfs:
        continue

    all_df = pd.concat(dfs, ignore_index=True)

    # Aggregate across sites, preserving MAF & depth
    agg = (all_df
           .groupby(['MAF', 'depth', 'Position'], as_index=False)['Num_Samples']
           .sum()
           .rename(columns={'Num_Samples': 'with_iSNV'}))
    agg['prop'] = agg['with_iSNV'] / tot
    agg['grp']  = grp
    scatter_dfs.append(agg)

if not scatter_dfs:
    raise RuntimeError("No scatter data found. Check files or column headers vs. pattern.")

combined_scatter = pd.concat(scatter_dfs, ignore_index=True)


### Plotting ###

plt.style.use("plot_style_settings.mplstyle")

ncols = len(depth_thresholds)
nrows = len(MAF_thresholds)

fig = plt.figure(figsize=(7.5, 6.5), dpi=600, constrained_layout=True)
gs = GridSpec(nrows, ncols, height_ratios=[1]*nrows, wspace=0.01, hspace=0.1, figure=fig)

# Only plot groups that have data
available_groups = sorted(combined_scatter['grp'].unique())
plot_groups = [g for g in group_order if g in available_groups]

# generate colors for groups
palette_scatter = sns.color_palette("tab10", n_colors=len(plot_groups))
group_colors = {g: palette_scatter[i] for i, g in enumerate(plot_groups)}


axes = [[None for _ in range(ncols)] for _ in range(nrows)]

for r, m in enumerate(MAF_thresholds):
    for c, d in enumerate(depth_thresholds):
        # share x within each column so genome position aligns
        share_with = axes[0][c] if r > 0 else None
        ax = fig.add_subplot(gs[r, c], sharex = share_with)

        for grp in plot_groups:
            sub = (combined_scatter
                   .query("grp == @grp and depth == @d and MAF == @m")
                   .sort_values("Position"))
            if not sub.empty:
                ax.scatter(
                    sub['Position'], sub['prop'],
                    s=1, marker='o', edgecolors='none',
                    color=group_colors[grp], alpha=0.7,
                    label=grp if (r == 0 and c == 0) else None
                )
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
        ax.minorticks_on()
        ax.grid(which="major", axis="both", linewidth=0.3, alpha=0.3)
        ax.grid(which="minor", axis="x", linewidth=0.3, alpha=0.1)
        ax.tick_params(axis="x", which="minor", length=0)
        min_reads = d * (m / 100.0)
        ax.text(
            0.5, 0.94, f"iSNV min. {fmt_reads(min_reads)} reads",
            transform=ax.transAxes, ha="center", fontsize=5,
            bbox= dict(facecolor='white', edgecolor='none', alpha=0.8, pad=0.8),
        )

        # Set titles and labels
        if r == 0:
            ax.set_title(f"Read depth ≥ {d}x", loc="center", pad=10)
        if c == 0:
            ax.set_ylabel("Proportion of samples")
        if c == len(depth_thresholds) - 1:
            ax.text(1.08, 0.5, f"MAF ≥ {m} %",
                    transform=ax.transAxes, fontsize=6, 
                    va="center")
        if r == nrows - 1:
            ax.set_xlabel("Genome position")

        axes[r][c] = ax

# Legend
handles, labels = axes[0][0].get_legend_handles_labels()
if handles:
    pairs = sorted(zip(labels, handles), key=lambda x: x[0])
    labels_sorted, handles_sorted = zip(*pairs)
    fig.legend(handles_sorted, labels_sorted, ncols = len(pairs), loc="upper center",
               markerscale=4, bbox_to_anchor=(0.5, 1.05))

plot_filename = 'figureS2_iSNV_scatter_depths.png'
plot_path = os.path.join(figures_dir, plot_filename)
plt.savefig(plot_path, format='png', bbox_inches='tight')
plt.close()

print(f"Figure saved as '{plot_path}'")
