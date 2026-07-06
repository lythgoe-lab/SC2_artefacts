import os
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import PercentFormatter, MultipleLocator


mask_file = "../../processed_data/10x/combined_masks_etc_for_upset.csv"
figures_dir = "../../figures"
primer_files = {
    "ARTIC_3": "../../ARTIC_3.primer.bed",
    "ARTIC_4": "../../ARTIC_4.primer.bed",
    "ARTIC_4.1": "../../ARTIC_4.1.primer.bed",
}
masks_to_plot = [
    "DeMaio_Mask", "DeMaio_Caution",
    "NORT_ARTIC_3", "NORT_ARTIC_4", "NORT_ARTIC_4.1", "NORW_ARTIC_4.1",
    "OXON_VeSeq", "PHEC_ARTIC_3", "SANG_ARTIC_3", "SANG_ARTIC_4.1"
]

# Get colormaps
cmap = plt.get_cmap('tab10')
site_colors = {mask: cmap(i) for i, mask in enumerate(masks_to_plot)}

# Define plot parameters
arrow_head_scale = 2
genome_windows = [(13500, 14500), (18500, 19500), (25000, 26000)]

mask_df = pd.read_csv(mask_file)
missing = set(masks_to_plot) - set(mask_df.columns)
if missing:
    raise KeyError(f"Missing columns in mask file: {missing}")

primer_data = {}
for scheme, path in primer_files.items():
    dfp = pd.read_csv(
        path, sep="\t", header=None, comment="#",
        usecols=[1,2,5], names=["start","end","strand"]
    ).astype({"start": int, "end": int})
    dfp = dfp[~((dfp.end < genome_windows[0][0]) |
                (dfp.start > genome_windows[-1][1]))].copy()
    primer_data[scheme] = dfp.sort_values('start').reset_index(drop=True)

mask_sites = [s for s in masks_to_plot if not s.startswith('DeMaio')]
all_pos = np.unique(
    np.concatenate([mask_df[s].dropna().astype(float).values
                    for s in mask_sites])
)

scheme_colors = {
    'ARTIC_3': 'royalblue',
    'ARTIC_4': 'teal',
    'ARTIC_4.1': 'forestgreen'
}

def load_sample_totals(threshold='10x'):
    file = '../../processed_data/samples_per_site_cov_summary.csv'
    if not os.path.exists(file):
        print(f"Error: {file} not found")
        sys.exit(1)
    df = pd.read_csv(file)
    df = df[df['Cov_Threshold'] == threshold].copy()
    df['site_short'] = df['Site'].str.replace(r'_ILLUMINA$', '', regex=True)
    return dict(zip(df['site_short'], df['Total_Samples']))

read_depths = pd.read_csv('../../processed_data/10x/read_depth_summary.csv')
sites = mask_sites.copy()
read_depths = read_depths[read_depths['Site'].isin(sites)]
sample_totals = load_sample_totals('10x')


# Plot
plt.style.use("plot_style_settings.mplstyle")

fig = plt.figure(figsize=(7.5, 8), constrained_layout=True)
gs = gridspec.GridSpec(
    4,
    len(genome_windows),
    height_ratios=[8, 3, 1, 3],
    wspace=0.01,
    hspace=0.05,
    figure=fig
)

# Top block (2 rows stacked)
gs_top = gridspec.GridSpecFromSubplotSpec(
    2, 1, subplot_spec=gs[0, :], height_ratios=[2, 2], hspace=0.0
)
ax_full_dots = fig.add_subplot(gs_top[0, 0])
ax_full_lines = fig.add_subplot(gs_top[1, 0], sharex=ax_full_dots)

mask_axes = [fig.add_subplot(gs[1, col]) for col in range(len(genome_windows))]
primer_axes = [fig.add_subplot(gs[2, col], sharex=mask_axes[col])
               for col in range(len(genome_windows))]
prop_axes = [fig.add_subplot(gs[3, col], sharex=mask_axes[col])
             for col in range(len(genome_windows))]

# Full-genome x-limits
genome_min = int(read_depths["Position"].min())
genome_max = int(read_depths["Position"].max())

# draw vertical guides at window edges to visually align with lower panels
window_edges = sorted({x for w in genome_windows for x in w})
for x in window_edges:
    ax_full_dots.axvline(x, color='gray', linestyle=':', linewidth=0.5, zorder=1)
    ax_full_lines.axvline(x, color='gray', linestyle=':', linewidth=0.5, zorder=1)

for site in sites:
    df_all = read_depths[read_depths["Site"] == site].copy()
    total = sample_totals.get(site, np.nan)
    if np.isnan(total):
        raise ValueError(f"No total-samples for {site!r}")
    prop_full = df_all["samples_depth_1000_no_gaps"] / total
    ax_full_lines.plot(
        df_all["Position"], prop_full,
        color=site_colors[site], linewidth=0.2, alpha=0.8,
        label=site.replace("NORW_ARTIC_4.1", "NORW ARTIC ?").replace("OXON_VeSeq", "OXON ve-SEQ").replace('_', ' ')
    )

ax_full_lines.set_xlim(genome_min, genome_max)
ax_full_lines.set_ylim(-0.05, 1.05)
ax_full_lines.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
ax_full_lines.tick_params(axis='y', labelleft=True, labelright=False)
ax_full_lines.xaxis.set_major_locator(MultipleLocator(2000))
ax_full_lines.xaxis.set_minor_locator(MultipleLocator(500)) 
ax_full_lines.tick_params(axis='x', which='both', rotation=30, pad=1, labelbottom=True)
ax_full_lines.set_xlabel('Genome position')
ax_full_lines.set_ylabel('Proportion samples\nread depth ≥ 1000x')
ax_full_lines.grid(False)

# Legend for the genome-wide lines
ax_full_lines.legend(
    loc='upper left',
    bbox_to_anchor=(-0.27, 0.82),
    fontsize='6',
    frameon=False,
    ncol=1
)

# Full-genome mask position scatter
mask_labels_full = [
    lbl.replace("NORW_ARTIC_4.1", "NORW ARTIC ?").replace("DeMaio", "De Maio").replace("OXON_VeSeq", "OXON ve-SEQ").replace("_", " ")
    for lbl in masks_to_plot
]
ax_full_dots.set_ylim(len(masks_to_plot) - 0.5, -0.5)
ax_full_dots.set_xlim(genome_min, genome_max)
ax_full_dots.set_yticks(range(len(masks_to_plot)))
ax_full_dots.set_yticklabels(mask_labels_full)
ax_full_dots.tick_params(axis='x', labelbottom=False)
for i, mask_name in enumerate(masks_to_plot):
    pos = mask_df[mask_name].dropna().astype(float).values
    if len(pos):
        ax_full_dots.scatter(
            pos, np.full_like(pos, i),
            s=2, alpha=0.75, color=site_colors[mask_name], linewidths=0, zorder=2
        )
ax_full_dots.grid(False)

mask_labels = [
    lbl.replace("NORW_ARTIC_4.1", "NORW ARTIC ?").replace("DeMaio", "De Maio").replace("OXON_VeSeq", "OXON ve-SEQ").replace("_", " ")
    for lbl in masks_to_plot
]

# Windows: Masks + Primers + Proportion ≥ 1000x
for col, (wmin, wmax) in enumerate(genome_windows):
    axm = mask_axes[col]
    axp = primer_axes[col]

    # Masks scatter
    for i, site in enumerate(masks_to_plot):
        pos = mask_df[site].dropna().astype(float)
        pos = pos[(pos >= wmin) & (pos <= wmax)]
        axm.scatter(pos, [i]*len(pos), s=0.3, alpha=0.6, color=site_colors[site])
    axm.set(ylim=(len(masks_to_plot)-0.5, -0.5),
            yticks=range(len(masks_to_plot)),
            yticklabels=mask_labels,
            xlim=(wmin, wmax))
    if col > 0:
        axm.tick_params(labelleft=False)
    axm.tick_params(labelbottom=False)
    axm.grid(False)

    # Primer arrows
    for idx, scheme in enumerate(primer_data):
        dfp = primer_data[scheme]
        levels, offsets = [], []
        for _, row in dfp.iterrows():
            placed = False
            for lvl, endpos in enumerate(levels):
                if row.start > endpos:
                    levels[lvl] = row.end
                    offsets.append(lvl)
                    placed = True
                    break
            if not placed:
                levels.append(row.end)
                offsets.append(len(levels)-1)
        for lvl, (_, row) in zip(offsets, dfp.iterrows()):
            y = idx + (lvl*0.2) - (len(levels)-1)*0.1
            kw = dict(
                arrowstyle='->',
                color=scheme_colors[scheme],
                lw=0.5,
                mutation_scale=arrow_head_scale,
                shrinkA=0, shrinkB=0
            )
            start, end = (row.start, row.end) if row.strand == '+' else (row.end, row.start)
            axp.annotate('', xy=(end, y), xytext=(start, y), arrowprops=kw)
    axp.set(ylim=(-0.5, len(primer_data)-0.5),
            yticks=range(len(primer_data)),
            yticklabels=[s.replace('ARTIC_', 'ARTIC ') for s in primer_data],
            xlim=(wmin, wmax))
    if col > 0:
        axp.tick_params(labelleft=False)
    axp.tick_params(axis='x', rotation=30, pad=1)
    for lbl, scheme in zip(axp.get_yticklabels(), primer_data):
        lbl.set_color(scheme_colors[scheme])

    # Overlap lines
    for scheme, dfp in primer_data.items():
        colr = scheme_colors.get(scheme, 'gray')
        for p in all_pos:
            if any((st <= p <= en) for st, en in zip(dfp.start, dfp.end)):
                axm.axvline(p, color=colr, linestyle=':', linewidth=0.3)
                axp.axvline(p, color=colr, linestyle=':', linewidth=0.3)

# shared x-labels on middle column
mid = len(genome_windows)//2
primer_axes[mid].set_xlabel('Genome position')
prop_axes[mid].set_xlabel('Genome position')

# Bottom: Samples with ≥ 1000× depth
for col, (wmin, wmax) in enumerate(genome_windows):
    axb = prop_axes[col]
    for site in sites:
        dfw = read_depths[
            (read_depths.Site == site) &
            (read_depths.Position >= wmin) &
            (read_depths.Position <= wmax)
        ]
        total = sample_totals.get(site, np.nan)
        if np.isnan(total):
            raise ValueError(f"No total-samples for {site!r}")
        prop = dfw['samples_depth_1000_no_gaps'] / total
        axb.plot(dfw.Position, prop,
                 label=site.replace("NORW_ARTIC_4.1", "NORW ARTIC ?").replace("OXON_VeSeq", "OXON ve-SEQ").replace('_', ' '),
                 color=site_colors[site],
                 linewidth=0.3)
    axb.set(xlim=(wmin, wmax), ylim=(-0.1, 1.1))
    axb.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    axb.tick_params(axis='x', rotation=30, pad=1)
    if col == 0:
        axb.set_ylabel("Proportion samples\nread depth ≥ 1000x")
    else:
        axb.tick_params(labelleft=False)
    axb.axhline(0, color='gray', linestyle=':', linewidth=0.2)
    axb.axhline(1, color='gray', linestyle=':', linewidth=0.2)

# single legend on bottom-left panel
prop_axes[0].legend(
    loc='upper left',
    bbox_to_anchor=(-0.9, 0.86),
    fontsize='6',
    frameon=False
)

# Align the left edges of the top plots to the bottom plots
fig.canvas.draw()
left_ref = prop_axes[0].get_position().x0
for ax in (ax_full_dots, ax_full_lines):
    pos = ax.get_position()
    # keep right edge, move left edge to reference
    new_width = pos.x1 - left_ref
    ax.set_position([left_ref, pos.y0, new_width, pos.height])


# add panel labels
fig.text(0.01, 0.99, "a", fontsize=8, fontweight="bold")
fig.text(0.01, 0.78, "b", fontsize=8, fontweight="bold")
fig.text(0.01, 0.53, "c", fontsize=8, fontweight="bold")
fig.text(0.01, 0.22, "d", fontsize=8, fontweight="bold")

plot_filename = 'figureS11_mask_positions_and_primer_depth_windows.pdf'
plot_path = os.path.join(figures_dir, "supplemental", plot_filename)
plt.savefig(plot_path, format='pdf', bbox_inches='tight', dpi=600)
plt.close()
print(f"Figure saved as '{plot_path}'")
