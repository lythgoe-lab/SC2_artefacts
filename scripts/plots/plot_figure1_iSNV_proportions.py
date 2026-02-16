import os
import re
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.ticker import ScalarFormatter, NullLocator
import matplotlib.ticker as mtick
import seaborn as sns
from matplotlib.patches import Rectangle


base_processed_dir = "../../processed_data/"
figures_dir = "../../figures/"
cov_folder = "10x"
depth_threshold = 1000

maf_thresholds = [3, 5, 10, 20]
threshold_labels = [f'MAF ≥{m}%' for m in maf_thresholds]

group_bins = [-0.5, 0.5, 5.5, 10.5, 15.5, 20.5, 25.5, np.inf]
group_labels = ['0', '1-5', '6-10', '11-15', '16-20', '21-25', '26+']

sharing_edges = [0.00, 0.01, 0.05, 0.10, 0.20, 0.50, 1.00]
sharing_labels = ['<1%', '1-5%', '5-10%', '10-20%', '20-50%', '>50%']

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

groups = {}
nort = [s for s in subset_sites if s.startswith("NORT")]
sang = [s for s in subset_sites if s.startswith("SANG")]
groups['NORT'] = nort
groups['SANG'] = sang
for site in subset_sites:
    if site not in nort + sang:
        key = site.split()[0]
        groups[key] = [site]

group_order1 = ["SANG", "NORW", "NORT", "PHEC", "OXON"]
group_order2 = sorted(group_order1)


# Process panel a: iSNVs per sample
threshold_pattern = re.compile(r'iSNVs_(\d+)%_depth_(\d+)')
all_long = []
for site in subset_sites:
    path = os.path.join(
        base_processed_dir, cov_folder,
        site_to_folder_map[site], 'total_iSNVs_per_sample.csv'
    )
    if not os.path.exists(path):
        continue

    df = pd.read_csv(path, low_memory=False)
    df['site'] = site
    non_thresh = ['site', 'ID']
    thresh_cols = [c for c in df.columns if c not in non_thresh]

    info = []
    for c in thresh_cols:
        m = threshold_pattern.match(c)
        if m:
            info.append({'col': c, 'MAF': int(m.group(1)), 'depth': int(m.group(2))})
    info = pd.DataFrame(info)

    long = df.melt(
        id_vars=non_thresh, value_vars=thresh_cols,
        var_name='col', value_name='iSNV_Count'
    )
    long = long.merge(info, on='col')
    long = long[long['depth'] == depth_threshold]
    all_long.append(long)

combined_data = pd.concat(all_long, ignore_index=True)

# Get total samples per group
def load_sample_totals(mapping):
    fn = os.path.join(base_processed_dir, 'samples_per_site_cov_summary.csv')
    df = pd.read_csv(fn)
    df = df[df['Cov_Threshold'] == cov_folder]
    totals = {}
    for grp, sites in mapping.items():
        total = 0
        for s in sites:
            row = df[df['Site'] == site_to_folder_map[s]].iloc[0]
            total += int(row['Total_Samples'])
        totals[grp] = total
    return totals

sample_totals = load_sample_totals(groups)

# Process panel b: shared iSNVs per group
pattern2 = re.compile(r"Samples_MAF_(?P<MAF>\d+)%_depth_(?P<depth>\d+)")
group_data = {}
for grp, sites in groups.items():
    dfs = []
    for s in sites:
        path = os.path.join(
            base_processed_dir, cov_folder,
            site_to_folder_map[s], 'samples_per_iSNV_position.csv'
        )
        if not os.path.exists(path):
            continue

        df = pd.read_csv(path)
        long = df.melt(id_vars=['Position'], var_name='Threshold', value_name='Num_Samples')
        parsed = long['Threshold'].str.extract(pattern2).astype(int)
        long['MAF']   = parsed['MAF']
        long['depth'] = parsed['depth']
        dfs.append(long)

    if not dfs:
        continue

    all_df = pd.concat(dfs, ignore_index=True)
    df_d = all_df[(all_df['depth'] == depth_threshold) & (all_df['Num_Samples'] > 0)].copy()
    total = sample_totals[grp]
    df_d['Prop'] = df_d['Num_Samples'] / total
    df_d['Bin']  = pd.cut(df_d['Prop'], bins=sharing_edges,
                         labels=sharing_labels, right=False)

    summary = (df_d[df_d['MAF'].isin(maf_thresholds)]
               .groupby(['Bin', 'MAF'], observed=True)
               .size().reset_index(name='Count'))
    group_data[grp] = summary

# Process panel c: scatter plot of shared iSNVs per position
scatter_maf_thresholds = [3, 10]
scatter_dfs = []
for grp, sites in groups.items():
    tot = sample_totals[grp]
    dfs = []
    for s in sites:
        path = os.path.join(
            base_processed_dir, cov_folder,
            site_to_folder_map[s], 'samples_per_iSNV_position.csv'
        )
        if not os.path.exists(path):
            continue

        df = pd.read_csv(path)
        long = df.melt(id_vars=['Position'], var_name='Threshold', value_name='Num_Samples')
        parsed = long['Threshold'].str.extract(pattern2).astype(int)
        long['MAF'] = parsed['MAF']
        long['depth'] = parsed['depth']

        sub = long[
            (long['MAF'].isin(scatter_maf_thresholds)) &
            (long['depth'] == depth_threshold)
        ].copy()
        dfs.append(sub)

    if not dfs:
        continue

    all_df = pd.concat(dfs, ignore_index=True)
    agg = (
        all_df
        .groupby(['MAF', 'Position'], as_index=False)['Num_Samples']
        .sum()
        .rename(columns={'Num_Samples': 'with_iSNV'})
    )
    agg['prop'] = agg['with_iSNV'] / tot
    agg['grp']  = grp
    scatter_dfs.append(agg)

combined_scatter = pd.concat(scatter_dfs, ignore_index=True)

orf_regions = [
    ("ORF1a", 266, 13483),
    ("ORF1b", 13468, 21555),
    ("S", 21563, 25384),
    ("3a", 25393, 26220),
    ("E", 26245, 26472),
    ("M", 26523, 27191),
    ("6", 27202, 27387),
    ("7a", 27394, 27759),
    ("7b", 27756, 27887),
    ("8", 27894, 28259),
    ("N", 28274, 29533),
    ("10", 29558, 29674),
]


palette_scatter = sns.color_palette("tab10", n_colors=len(group_order1))
group_colors = {g: palette_scatter[i] for i, g in enumerate(group_order1)}

orf_palette = sns.color_palette("Pastel1", n_colors=len(orf_regions))
orf_colors = {orf: orf_palette[i] for i, (orf, _, _) in enumerate(orf_regions)}



##### Plotting ######

plt.style.use("plot_style_settings.mplstyle")

# Set up the figure and gridspec
ncols = len(group_order1)

fig = plt.figure(figsize=(7, 9), dpi=600)
outer = GridSpec(3, ncols,
                 height_ratios=[0.2, 0.2, 0.6],
                 hspace=0.35,
                 figure=fig)
inner = GridSpecFromSubplotSpec(2, 1,
                                height_ratios=[1, 1],
                                hspace=0.25,
                                subplot_spec=outer[2, :])



# Panel a: iSNV counts per sample
axes_prop = []
first_prop_ax = None
colors_prop = sns.color_palette('rocket', n_colors=len(group_labels))

for idx, grp in enumerate(group_order2):
    ax = fig.add_subplot(outer[0, idx], sharey=first_prop_ax)
    if first_prop_ax is None:
        first_prop_ax = ax

    gd = combined_data[combined_data['site'].isin(groups[grp])]

    props = []
    for m in maf_thresholds:
        td = gd[gd['MAF'] == m].dropna(subset=['iSNV_Count'])
        counts = (pd.cut(td['iSNV_Count'], bins=group_bins, labels=group_labels)
                  .value_counts().reindex(group_labels, fill_value=0))
        props.append((counts / counts.sum()).values)
    props = np.array(props).T

    x = np.arange(len(maf_thresholds))
    bottom = np.zeros_like(x, dtype=float)
    for i in range(len(group_labels)):
        ax.bar(
            x,
            props[i],
            bottom=bottom,
            color=colors_prop[i],
            label=group_labels[i] if idx == 0 else None
        )
        bottom += props[i]

    ax.set_title(f'{grp}\n(n={sample_totals[grp]})')
    ax.set_xticks(x)
    ax.set_xticklabels(threshold_labels, rotation=45)
    if idx == 0:
        ax.set_ylabel('Proportion of samples')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
    axes_prop.append(ax)

# hide y‐tick labels on non‐left panels
for ax in axes_prop[1:]:
    ax.tick_params(labelleft=False)

# legend for panel a
h, l = axes_prop[0].get_legend_handles_labels()
h = h[::-1]
l = l[::-1]
fig.legend(
    h, l,
    title='iSNVs per sample',
    loc='upper right',
    bbox_to_anchor=(0.99, 0.95),
    frameon=False
)


# Panel b: shared iSNV proportions per group
axes_cnt = []
first_cnt_ax = None
palette_b = sns.color_palette('rocket', n_colors=len(sharing_labels))
bar_w = 0.8 / len(sharing_labels)
yticks = [1, 2, 5, 10, 50, 100, 500, 1000, 5000, 10000, 30000]

for idx, grp in enumerate(group_order2):
    ax = fig.add_subplot(outer[1, idx], sharey=first_cnt_ax)
    if first_cnt_ax is None:
        first_cnt_ax = ax

    x2 = np.arange(len(maf_thresholds))
    for i, b in enumerate(sharing_labels):
        counts = [
            group_data[grp].loc[
                (group_data[grp]['Bin'] == b) & (group_data[grp]['MAF'] == m),
                'Count'
            ].sum()
            for m in maf_thresholds
        ]
        offs = x2 + (i - (len(sharing_labels) - 1) / 2) * bar_w
        ax.bar(
            offs,
            counts,
            width = bar_w,
            color = palette_b[i],
            label = b if idx == 0 else None
        )

    ax.set_title(f'{grp}\n(n={sample_totals[grp]})')
    ax.set_xticks(x2)
    ax.set_xticklabels(threshold_labels, rotation=45)

    ax.set_yscale('log')
    ax.set_ylim(1, yticks[-1] * 1.1)
    ax.set_yticks(yticks)
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.grid(axis='y', alpha=0.3, linewidth=0.5)
    if idx == 0:
        ax.set_ylabel('Number of iSNV positions\n(log scale)')

    axes_cnt.append(ax)

# hide y‐tick labels on non‐left panels
for ax in axes_cnt[1:]:
    ax.tick_params(labelleft=False)

# legend for panel b
h2, l2 = axes_cnt[0].get_legend_handles_labels()
h2 = h2[::-1]
l2 = l2[::-1]
fig.legend(
    h2, l2,
    loc='upper right',
    bbox_to_anchor=(1, 0.73),
    frameon=False,
    title = 'Proportion of samples\n sharing iSNV positions'
)


# Panel c: scatter plot of shared iSNVs per position
# Row 1: MAF ≥ 3%
ax3 = fig.add_subplot(inner[0, :])
for grp in group_order1:
    sub = combined_scatter.query("grp == @grp and MAF == 3").sort_values("Position")
    ax3.scatter(
        sub.Position,
        sub.prop,
        s=1.5,
        marker='o',
        edgecolors='none',
        color=group_colors[grp],
        alpha=0.7,
        label=grp
    )
ax3.set_ylim(0,1)
ax3.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
ax3.set_ylabel("Proportion of samples\nwith iSNV")
ax3.set_title("MAF ≥ 3 %", loc="center")
ax3.minorticks_on()
ax3.grid(which="major", axis="both", linewidth=0.4, alpha=0.3)
ax3.grid(which="minor", axis="x", linewidth=0.4, alpha=0.1)
ax3.tick_params(axis="x", which="minor", length=0)
ax3.axhline(0.015, linestyle="--", linewidth=0.5, color="gray")

# Row 2: MAF ≥ 10%
ax10 = fig.add_subplot(inner[1, :], sharex=ax3)
for grp in group_order1:
    sub = combined_scatter.query("grp == @grp and MAF == 10").sort_values("Position")
    ax10.scatter(
        sub.Position,
        sub.prop,
        s=1.5,
        marker='o',
        edgecolors='none',
        color=group_colors[grp],
        alpha=0.7,
        label=grp
    )
ax10.set_ylim(0, 1)
ax10.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
ax10.set_ylabel("Proportion of samples\nwith iSNV")
ax10.set_title("MAF ≥ 10 %", loc="center")
ax10.minorticks_on()
ax10.grid(which="major", axis="both", linewidth=0.4, alpha=0.3)
ax10.grid(which="minor", axis="x", linewidth=0.4, alpha=0.1)
ax10.tick_params(axis="x", which="minor", length=0)
ax10.axhline(0.015, linestyle="--", linewidth=0.5, color="gray")
ax10.set_xlabel("Genome position")

# legend for panel c
handles, labels = ax3.get_legend_handles_labels()
pair_list = sorted(zip(labels, handles), key=lambda x: x[0])
sorted_labels, sorted_handles = zip(*pair_list)
fig.legend(
    sorted_handles, sorted_labels,
    loc="upper right",
    markerscale=4,
    bbox_to_anchor=(0.97, 0.36)
)

# Add ORF bars below the bottom panel
y_frac = -0.4
bar_frac = 0.1
leader_len = 0.02
label_pad = 0.01

# Choose placement of some ORF labels outside the bar
external_orfs = {"E", "6", "7a", "7b", "10"}
external_side = {
    "E": "above",
    "6": "below",
    "7a": "above",
    "7b": "below",
    "10": "above",
}

for orf, start, end in orf_regions:
    width = (end - start) + 1

    # Draw the bar
    rect = Rectangle(
        (start, y_frac),
        width,
        bar_frac,
        transform=ax10.get_xaxis_transform(),
        facecolor=orf_colors[orf],
        alpha=0.4,
        edgecolor="black",
        linewidth=0.5,
        clip_on=False,
        zorder=1,
    )
    ax10.add_patch(rect)

    # Center x of this ORF
    x_mid = start + width / 2.0

    if orf in external_orfs:
        side = external_side.get(orf, "above")
        if side == "above":
            y0 = y_frac + bar_frac
            y1 = y0 + leader_len
            text_y = y1 + label_pad
            va = "bottom"
        else:
            y0 = y_frac
            y1 = y0 - leader_len
            text_y = y1 - label_pad
            va = "top"

        # thin vertical leader line
        ax10.plot(
            [x_mid, x_mid], [y0, y1],
            transform=ax10.get_xaxis_transform(),
            color="black",
            linewidth=0.4,
            solid_capstyle="butt",
            zorder=2,
            clip_on=False
        )

        # external label
        ax10.text(
            x_mid, text_y, orf,
            transform=ax10.get_xaxis_transform(),
            ha="center", va=va,
            fontsize=5,
            zorder=3,
            clip_on=False
        )
    else:
        # internal label
        ax10.text(
            x_mid,
            y_frac + bar_frac / 2.0,
            orf,
            ha="center",
            va="center",
            fontsize=5,
            transform=ax10.get_xaxis_transform(),
            clip_on=False,
            zorder=2
        )

# Panel labels
fig.text(0.01, 0.99, 'a', fontweight='bold', fontsize=8)
fig.text(0.01, 0.76, 'b', fontweight='bold', fontsize=8)
fig.text(0.01, 0.53, 'c', fontweight='bold', fontsize=8)

plt.subplots_adjust(left=0.10, bottom=0.10, right=0.83, top=0.95)

plot_filename = 'figure1_iSNV_proportions.pdf'
plot_path = os.path.join(figures_dir, plot_filename)
plt.savefig(plot_path, format='pdf', bbox_inches='tight')
plt.close()

print(f"Figure saved as '{plot_path}'")