import matplotlib as mpl
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import os
import matplotlib.ticker as mtick
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

base_path = '../../processed_data'
figure_path = '../../figures/supplemental'
cov_folder = '10x'
depth_threshold = 1000
maf_thresholds  = list(range(2, 21))

summary_csv = os.path.join(base_path, 'samples_per_site_cov_summary.csv')

site_to_folder_map = {
    "OXON ve-SEQ": "OXON_VeSeq_ILLUMINA",
    "NORT ARTIC 3": "NORT_ARTIC_3_ILLUMINA",
    "NORT ARTIC 4": "NORT_ARTIC_4_ILLUMINA",
    "NORT ARTIC 4.1": "NORT_ARTIC_4.1_ILLUMINA",
    "NORW ARTIC ?": "NORW_ARTIC_4.1_ILLUMINA",
    "PHEC ARTIC 3": "PHEC_ARTIC_3_ILLUMINA",
    "SANG ARTIC 3": "SANG_ARTIC_3_ILLUMINA",
    "SANG ARTIC 4.1": "SANG_ARTIC_4.1_ILLUMINA"
}
subset_sites = [
    "OXON ve-SEQ", "NORT ARTIC 3", "NORT ARTIC 4", 
    "NORT ARTIC 4.1", "NORW ARTIC ?",
    "PHEC ARTIC 3", "SANG ARTIC 3", "SANG ARTIC 4.1"
]

# Color mapping
cmap = plt.get_cmap('tab10')
colors = cmap(np.linspace(0, 1, len(subset_sites)))
site_colors = {site: color for site, color in zip(subset_sites, colors)}


# Helper function to style violin “bodies” and “tails”
def style_violin_parts(vp, face_colors, alpha=0.3, edge_width=0.2):
    for i, c in enumerate(face_colors):
        body = vp['bodies'][i]
        body.set_facecolor(c)
        body.set_edgecolor('black')
        body.set_alpha(alpha)
        body.set_linewidth(edge_width)
    for part in ('cmins', 'cmaxes'):
        vp[part].set_edgecolor('black')
        vp[part].set_alpha(alpha)
        vp[part].set_linewidth(edge_width)
    vp['cbars'].set_color('black')
    vp['cbars'].set_alpha(alpha)
    vp['cbars'].set_linewidth(edge_width)


# Load iSNV counts
all_long_data_counts = []
threshold_pattern_counts = re.compile(r'iSNVs_(\d+)%_depth_(\d+)x')

for site_name in subset_sites:
    lab_folder = site_to_folder_map[site_name]
    file_path = os.path.join(base_path, cov_folder, lab_folder, 'total_iSNVs_per_sample.csv')
    if not os.path.exists(file_path):
        print(f"[Warning] File not found: {file_path}. Skipping.")
        continue

    data = pd.read_csv(file_path, low_memory=False)
    data['site'] = site_name

    non_threshold_cols = ['site', 'ID']
    threshold_cols = [c for c in data.columns if c not in non_threshold_cols]

    def parse_threshold_counts(col_name):
        match = threshold_pattern_counts.match(col_name)
        if match:
            return pd.Series({'MAF': int(match.group(1)), 'read_depth': int(match.group(2))})
        return pd.Series({'MAF': None, 'read_depth': None})

    threshold_info = pd.DataFrame([parse_threshold_counts(c) for c in threshold_cols])
    threshold_info['Threshold_Column'] = threshold_cols
    threshold_info.dropna(subset=['MAF', 'read_depth'], inplace=True)
    threshold_info.reset_index(drop=True, inplace=True)

    long_data = data.melt(
        id_vars=non_threshold_cols,
        value_vars=threshold_cols,
        var_name='Threshold_Column',
        value_name='iSNV_Count'
    ).merge(threshold_info, on='Threshold_Column', how='left')
    long_data.dropna(subset=['MAF', 'read_depth', 'iSNV_Count'], inplace=True)
    long_data['MAF'] = long_data['MAF'].astype(int)
    long_data['read_depth'] = long_data['read_depth'].astype(int)

    all_long_data_counts.append(long_data)

if not all_long_data_counts:
    raise FileNotFoundError("No data files for iSNV counts were found.")
combined_counts = pd.concat(all_long_data_counts, ignore_index=True)
combined_counts = combined_counts[combined_counts['read_depth'] == depth_threshold]

if not os.path.exists(summary_csv):
    raise FileNotFoundError(f"[Error] Cannot find summary CSV: {summary_csv}")
df_summary = pd.read_csv(summary_csv, low_memory=False)
df_summary = df_summary[df_summary['Cov_Threshold'] == cov_folder]

site_total_samples = {}
for site_name in subset_sites:
    folder_name = site_to_folder_map[site_name]
    row = df_summary[df_summary['Site'] == folder_name]
    if row.empty:
        raise KeyError(f"[Error] No row in summary CSV for site-folder '{folder_name}'")
    site_total_samples[site_name] = int(row.iloc[0]['Total_Samples'])

# Load proportion-sharing
all_long_data_prop = []
threshold_pattern_prop = re.compile(r"Samples_MAF_(\d+)%_depth_(\d+)x")

for site_name in subset_sites:
    lab_folder = site_to_folder_map[site_name]
    file_path = os.path.join(base_path, cov_folder, lab_folder, 'samples_per_iSNV_position.csv')
    if not os.path.exists(file_path):
        print(f"[Warning] File not found: {file_path}. Skipping {site_name}.")
        continue

    data = pd.read_csv(file_path, low_memory=False)
    data['site'] = site_name

    non_threshold_cols = ['site', 'Position']
    threshold_cols = [c for c in data.columns if c not in non_threshold_cols]

    long_data = data.melt(
        id_vars=non_threshold_cols,
        value_vars=threshold_cols,
        var_name='Threshold_Column',
        value_name='Num_Samples'
    )
    parsed = long_data['Threshold_Column'].str.extract(threshold_pattern_prop).astype(float)
    parsed.columns = ['MAF', 'read_depth']
    long_data['MAF'] = parsed['MAF'].astype(int)
    long_data['read_depth'] = parsed['read_depth'].astype(int)
    long_data.dropna(subset=['MAF', 'read_depth'], inplace=True)

    ld = long_data[(long_data['read_depth'] == depth_threshold) & (long_data['Num_Samples'] > 0)].copy()
    denom = site_total_samples.get(site_name)
    if denom is None:
        raise KeyError(f"[Error] No total-samples recorded for site '{site_name}'")
    ld['Prop'] = ld['Num_Samples'] / denom
    all_long_data_prop.append(ld)

if not all_long_data_prop:
    raise FileNotFoundError("No data files for proportion-sharing were found.")
combined_prop = pd.concat(all_long_data_prop, ignore_index=True)
combined_prop = combined_prop[combined_prop['read_depth'] == depth_threshold]


### Plotting ###

plt.style.use("plot_style_settings.mplstyle")

fig = plt.figure(figsize=(8, 8))
gs = GridSpec(
    11, 10,
    height_ratios=[0.7, 3, 0.2, 0.7, 3, 0.5, 0.7, 3, 0.2, 0.7, 3],
    hspace=0.2, wspace=0.1
)

# Create axes
ax_counts_high_1, ax_counts_low_1 = [], []
ax_counts_high_2, ax_counts_low_2 = [], []
ax_props_high_1, ax_props_low_1 = [], []
ax_props_high_2, ax_props_low_2 = [], []

# split MAF thresholds in two chunks of 10 for plotting two rows
maf_chunk1 = maf_thresholds[:10]
maf_chunk2 = maf_thresholds[10:]

for col in range(10):
    ax_counts_high_1.append(fig.add_subplot(gs[0, col]))
    ax_counts_low_1.append(fig.add_subplot(gs[1, col], sharex=ax_counts_high_1[-1]))
    ax_counts_high_2.append(fig.add_subplot(gs[3, col]))
    ax_counts_low_2.append(fig.add_subplot(gs[4, col], sharex=ax_counts_high_2[-1]))

    ax_props_high_1.append(fig.add_subplot(gs[6, col]))
    ax_props_low_1.append(fig.add_subplot(gs[7, col], sharex=ax_props_high_1[-1]))
    ax_props_high_2.append(fig.add_subplot(gs[9, col]))
    ax_props_low_2.append(fig.add_subplot(gs[10, col], sharex=ax_props_high_2[-1]))


# set Y-axis limits
count_low_max = 200
count_high_min = 201
count_high_max = 10000

prop_low_max = 0.20
prop_high_min = 0.21
prop_high_max = 1.0


# Shared styling for count panels
for axes in [(ax_counts_high_1, ax_counts_low_1), (ax_counts_high_2, ax_counts_low_2)]:
    for a_high, a_low in zip(*axes):
        a_high.tick_params(axis='x', bottom=False, labelbottom=False)
        a_low.tick_params(axis='x', bottom=True)
        a_high.set_ylim(count_high_min, count_high_max)
        a_low.set_ylim(0, count_low_max)

        a_high.spines['bottom'].set_linestyle('--')
        a_high.spines['bottom'].set_color('grey')
        a_high.spines['bottom'].set_linewidth(0.3)
        a_low.spines['top'].set_linestyle('--')
        a_low.spines['top'].set_color('grey')
        a_low.spines['top'].set_linewidth(0.3)

        a_high.set_yticks([201, 5000, 10000])
        a_high.set_yticks([2600, 7500], minor=True)
        a_low.yaxis.set_minor_locator(mtick.MultipleLocator(10))

        a_high.tick_params(axis='y', labelleft=True)
        a_low.tick_params(axis='y', labelleft=True)

# Shared styling for proportion panels
for axes in [(ax_props_high_1, ax_props_low_1), (ax_props_high_2, ax_props_low_2)]:
    for p_high, p_low in zip(*axes):
        p_high.tick_params(axis='x', bottom=False, labelbottom=False)
        p_low.tick_params(axis='x', bottom=True)
        p_high.set_ylim(prop_high_min, prop_high_max)
        p_low.set_ylim(0, prop_low_max)

        p_high.spines['bottom'].set_linestyle('--')
        p_high.spines['bottom'].set_color('grey')
        p_high.spines['bottom'].set_linewidth(0.3)
        p_low.spines['top'].set_linestyle('--')
        p_low.spines['top'].set_color('grey')
        p_low.spines['top'].set_linewidth(0.3)

        p_high.set_yticks([0.21, 0.6, 1.0])
        p_high.set_yticks([0.4, 0.8], minor=True)
        p_low.yaxis.set_minor_locator(mtick.MultipleLocator(0.01))

        p_high.tick_params(axis='y', labelleft=True)
        p_low.tick_params(axis='y', labelleft=True)


# Panel a: counts - first chunk of MAFs
for col, maf in enumerate(maf_chunk1):
    a_high = ax_counts_high_1[col]
    a_low = ax_counts_low_1[col]

    data_maf = combined_counts[combined_counts['MAF'] == maf]
    count_by_site = [data_maf[data_maf['site'] == site]['iSNV_Count'].values for site in subset_sites]

    vp_h = a_high.violinplot(count_by_site, positions=np.arange(1, len(subset_sites) + 1),
                             showmedians=False, bw_method=1.0)
    vp_l = a_low.violinplot(count_by_site, positions=np.arange(1, len(subset_sites) + 1),
                             showmedians=False, bw_method=1.0)
    face_colors = [site_colors[s] for s in subset_sites]
    style_violin_parts(vp_h, face_colors=face_colors)
    style_violin_parts(vp_l, face_colors=face_colors)

    a_high.boxplot(count_by_site, positions=np.arange(1, len(subset_sites) + 1),
                   widths=0.2, showcaps=True, showfliers=False, medianprops={'color': 'black'})
    a_low.boxplot(count_by_site, positions=np.arange(1, len(subset_sites) + 1),
                  widths=0.2, showcaps=True, showfliers=False, medianprops={'color': 'black'})

    for i, site in enumerate(subset_sites):
        vals = count_by_site[i]
        if len(vals) == 0:
            continue
        low_mask = vals <= count_low_max
        high_mask = vals >= count_high_min
        a_low.scatter(np.random.normal(i + 1, 0.08, low_mask.sum()), vals[low_mask],
                      color=site_colors[site], alpha=0.3, s=0.1)
        a_high.scatter(np.random.normal(i + 1, 0.08, high_mask.sum()), vals[high_mask],
                       color=site_colors[site], alpha=0.3, s=0.1)

    a_low.tick_params(axis='x', pad=0.5)
    a_high.set_title(f"MAF ≥ {maf}%", pad=4, fontsize=6)
    if col == 0:
        a_low.set_ylabel("iSNVs per sample")

# Panel a: counts – next 10 MAFs
for col, maf in enumerate(maf_chunk2):
    a_high = ax_counts_high_2[col]
    a_low = ax_counts_low_2[col]

    data_maf = combined_counts[combined_counts['MAF'] == maf]
    count_by_site = [data_maf[data_maf['site'] == site]['iSNV_Count'].values for site in subset_sites]

    vp_h = a_high.violinplot(count_by_site, positions=np.arange(1, len(subset_sites) + 1),
                             showmedians=False, bw_method=1.0)
    vp_l = a_low.violinplot(count_by_site, positions=np.arange(1, len(subset_sites) + 1),
                             showmedians=False, bw_method=1.0)
    style_violin_parts(vp_h, face_colors=face_colors)
    style_violin_parts(vp_l, face_colors=face_colors)

    a_high.boxplot(count_by_site, positions=np.arange(1, len(subset_sites) + 1),
                   widths=0.2, showcaps=True, showfliers=False, medianprops={'color': 'black'})
    a_low.boxplot(count_by_site, positions=np.arange(1, len(subset_sites) + 1),
                  widths=0.2, showcaps=True, showfliers=False, medianprops={'color': 'black'})

    for i, site in enumerate(subset_sites):
        vals = count_by_site[i]
        if len(vals) == 0:
            continue
        low_mask = vals <= count_low_max
        high_mask = vals >= count_high_min
        a_low.scatter(np.random.normal(i + 1, 0.08, low_mask.sum()), vals[low_mask],
                      color=site_colors[site], alpha=0.3, s=0.1)
        a_high.scatter(np.random.normal(i + 1, 0.08, high_mask.sum()), vals[high_mask],
                       color=site_colors[site], alpha=0.3, s=0.1)

    a_low.tick_params(axis='x', pad=0.5)
    a_high.set_title(f"MAF ≥ {maf}%", pad=4, fontsize=6)
    if col == 0:
        a_low.set_ylabel("iSNVs per sample")

# Panel b: proportions – first chunk of MAFs
for col, maf in enumerate(maf_chunk1):
    p_high = ax_props_high_1[col]
    p_low = ax_props_low_1[col]

    data_maf = combined_prop[combined_prop['MAF'] == maf]
    prop_by_site = [data_maf[data_maf['site'] == site]['Prop'].values for site in subset_sites]

    vp_h = p_high.violinplot(prop_by_site, positions=np.arange(1, len(subset_sites) + 1),
                             showmedians=False, bw_method=1.0)
    vp_l = p_low.violinplot(prop_by_site, positions=np.arange(1, len(subset_sites) + 1),
                             showmedians=False, bw_method=1.0)
    style_violin_parts(vp_h, face_colors=face_colors, alpha=0.3, edge_width=0.1)
    style_violin_parts(vp_l, face_colors=face_colors, alpha=0.3, edge_width=0.1)

    p_high.boxplot(prop_by_site, positions=np.arange(1, len(subset_sites) + 1),
                   widths=0.2, showcaps=True, showfliers=False, medianprops={'color': 'black'})
    p_low.boxplot(prop_by_site, positions=np.arange(1, len(subset_sites) + 1),
                  widths=0.2, showcaps=True, showfliers=False, medianprops={'color': 'black'})

    for i, site in enumerate(subset_sites):
        vals = prop_by_site[i]
        if len(vals) == 0:
            continue
        low_mask = vals <= prop_low_max
        high_mask = vals >= prop_high_min
        p_low.scatter(np.random.normal(i + 1, 0.08, low_mask.sum()), vals[low_mask],
                      color=site_colors[site], alpha=0.3, s=0.1)
        p_high.scatter(np.random.normal(i + 1, 0.08, high_mask.sum()), vals[high_mask],
                       color=site_colors[site], alpha=0.3, s=0.1)

    p_low.tick_params(axis='x', pad=0.5)
    p_high.set_title(f"MAF ≥ {maf}%", pad=4, fontsize=6)
    if col == 0:
        p_low.set_ylabel("Proportion shared\niSNV positions")
        p_low.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
        p_high.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))

# Panel b: proportions – next 10 MAFs
for col, maf in enumerate(maf_chunk2):
    p_high = ax_props_high_2[col]
    p_low = ax_props_low_2[col]

    data_maf = combined_prop[combined_prop['MAF'] == maf]
    prop_by_site = [data_maf[data_maf['site'] == site]['Prop'].values for site in subset_sites]

    vp_h = p_high.violinplot(prop_by_site, positions=np.arange(1, len(subset_sites) + 1),
                             showmedians=False, bw_method=1.0)
    vp_l = p_low.violinplot(prop_by_site, positions=np.arange(1, len(subset_sites) + 1),
                             showmedians=False, bw_method=1.0)
    style_violin_parts(vp_h, face_colors=face_colors, alpha=0.3, edge_width=0.1)
    style_violin_parts(vp_l, face_colors=face_colors, alpha=0.3, edge_width=0.1)

    p_high.boxplot(prop_by_site, positions=np.arange(1, len(subset_sites) + 1),
                   widths=0.2, showcaps=True, showfliers=False, medianprops={'color': 'black'})
    p_low.boxplot(prop_by_site, positions=np.arange(1, len(subset_sites) + 1),
                  widths=0.2, showcaps=True, showfliers=False, medianprops={'color': 'black'})

    for i, site in enumerate(subset_sites):
        vals = prop_by_site[i]
        if len(vals) == 0:
            continue
        low_mask = vals <= prop_low_max
        high_mask = vals >= prop_high_min
        p_low.scatter(np.random.normal(i + 1, 0.08, low_mask.sum()), vals[low_mask],
                      color=site_colors[site], alpha=0.3, s=0.1)
        p_high.scatter(np.random.normal(i + 1, 0.08, high_mask.sum()), vals[high_mask],
                       color=site_colors[site], alpha=0.3, s=0.1)

    p_low.tick_params(axis='x', pad=0.5)
    p_high.set_title(f"MAF ≥ {maf}%", pad=4, fontsize=6)
    if col == 0:
        p_low.set_ylabel("Proportion shared\niSNV positions")
        p_low.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
        p_high.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))

# Add legend
legend_handles = [
    Line2D([0], [0], color = site_colors[s], lw = 4, label = s)
    for s in subset_sites
]
fig.legend(
    handles = legend_handles, 
    loc = 'upper center',
    bbox_to_anchor = (0.5, 0.95),
    ncols = len(subset_sites)/2,
    frameon = False,
    fontsize = 6
)

# Clean up axes
for row_axes in [
    (ax_counts_high_1, ax_counts_low_1),
    (ax_counts_high_2, ax_counts_low_2),
    (ax_props_high_1, ax_props_low_1),
    (ax_props_high_2, ax_props_low_2)
]:
    for ax_list in row_axes:
        for i, ax in enumerate(ax_list):
            # Remove all x-axis ticks and labels
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.tick_params(axis='x', length=0)

            # Remove y-axis labels/ticks unless leftmost
            if i != 0:
                ax.set_yticklabels([])

# Remove empty subplots
for col in range(len(maf_chunk1), 10):
    ax_counts_high_1[col].axis('off')
    ax_counts_low_1[col].axis('off')
    ax_props_high_1[col].axis('off')
    ax_props_low_1[col].axis('off')
for col in range(len(maf_chunk2), 10):
    ax_counts_high_2[col].axis('off')
    ax_counts_low_2[col].axis('off')
    ax_props_high_2[col].axis('off')
    ax_props_low_2[col].axis('off')

# Panel labels
fig.text(0.05, 0.90, 'a', fontweight='bold', fontsize=8)
fig.text(0.05, 0.50, 'b', fontweight='bold', fontsize=8)

plt.tight_layout()

outname = f"figureS3_violin_boxplots_iSNV_counts_and_shared.png"
outpath = os.path.join(figure_path, outname)

plt.savefig(outpath, bbox_inches="tight", format ='png')
plt.close()
print(f"Saved {outpath}")