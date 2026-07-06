import os
import re
import ast
import numpy as np
import pandas as pd
import dask.dataframe as dd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mtick
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from math import ceil
from collections import Counter, defaultdict
import seaborn as sns
import matplotlib.lines as mlines


base_processed_dir = "../../processed_data/"
figures_dir = "../../figures/supplemental"

site_to_folder_map = {
    'NORT ARTIC 3': 'NORT_ARTIC_3_ILLUMINA',
    'NORT ARTIC 4': 'NORT_ARTIC_4_ILLUMINA',
    'NORT ARTIC 4.1': 'NORT_ARTIC_4.1_ILLUMINA',
    'NORW ARTIC ?': 'NORW_ARTIC_4.1_ILLUMINA',
    'OXON ve-SEQ': 'OXON_VeSeq_ILLUMINA',
    'PHEC ARTIC 3': 'PHEC_ARTIC_3_ILLUMINA',
    'SANG ARTIC 3': 'SANG_ARTIC_3_ILLUMINA',
    'SANG ARTIC 4.1': 'SANG_ARTIC_4.1_ILLUMINA',
}
sites = [
    "NORT ARTIC 3",
    "NORT ARTIC 4",
    "NORT ARTIC 4.1",
    "NORW ARTIC ?",
    "OXON ve-SEQ",
    "PHEC ARTIC 3",
    "SANG ARTIC 3",
    "SANG ARTIC 4.1",
]
def site_key(name: str) -> str:
    return site_to_folder_map[name].replace('_ILLUMINA', '')

all_mafs = list(range(2, 21))

# panel a: proportion of mask schemes with masked positions
cov_folder  = "10x"
mask_folder = "mask_files_depth_1000x"
pattern = re.compile(r'(\d+(?:\.\d+)?)%shared_(\d+)%MAF\.csv')

def load_masking_data(folder, shared_pct, maf_pct):
    path = os.path.join(base_processed_dir, cov_folder, folder, mask_folder,
                        f"{shared_pct}%shared_{maf_pct}%MAF.csv")
    if not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            return [int(float(line.strip())) for line in f if line.strip()]
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return []

records = []
for site in sites:
    folder = site_to_folder_map.get(site)
    if not folder:
        continue
    dir_path = os.path.join(base_processed_dir, cov_folder, folder, mask_folder)
    if not os.path.isdir(dir_path):
        continue
    for fname in os.listdir(dir_path):
        m = pattern.match(fname)
        if not m:
            continue
        shared_pct, maf_pct = float(m.group(1)), int(m.group(2))
        positions = load_masking_data(folder, shared_pct, maf_pct)
        records.append({
            "site": site,
            "mask": fname.replace('.csv',''),
            "shared_pct": shared_pct,
            "maf_pct": maf_pct,
            "masked_positions": positions
        })

df_fs = pd.DataFrame(records)

# masks count per site
masks_per_site = df_fs.groupby("site")["mask"].nunique().to_dict()

# per-site position frequencies
site_position_freq = {}
for site in sites:
    df_site = df_fs[df_fs["site"] == site]
    total_masks = masks_per_site.get(site, 0)
    pos_counts = defaultdict(int)
    for pos_list in df_site["masked_positions"]:
        for pos in pos_list:
            pos_counts[pos] += 1

    sorted_pos = sorted(pos_counts.keys())
    proportions = [pos_counts[p] / total_masks for p in sorted_pos] if total_masks else []
    site_position_freq[site] = (sorted_pos, proportions)


# Load mask files for panel b and c
mask_path = os.path.join(base_processed_dir, '10x/final_masks_all_centres.csv')
mask_df = pd.read_csv(mask_path)

mask_df = mask_df[
    mask_df['mask'].str.startswith('1.5%shared_') |
    mask_df['mask'].str.startswith('positions_in_over_')
].copy()

# parse positions column once
mask_df['positions'] = mask_df['positions'].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else []
)
mask_df['Masked_Count'] = mask_df['positions'].apply(len)

# split to identify adaptive/prevalent labels
adaptive_df = mask_df[mask_df['mask'].str.startswith('1.5%shared_')].copy()
prevalent_df = mask_df[mask_df['mask'].str.startswith('positions_in_over_')].copy()

for df, name in [(adaptive_df, "adaptive"), (prevalent_df, "prevalent")]:
    dupes = df['Centre_protocol'][df['Centre_protocol'].duplicated()]
    if not dupes.empty:
        raise RuntimeError(f"Found multiple {name} masks for: {set(dupes)}")

adaptive_masks = adaptive_df.set_index('Centre_protocol')['mask'].to_dict()
prevalent_masks = prevalent_df.set_index('Centre_protocol')['mask'].to_dict()


base_sites = [site_key(s) for s in sites]

# Panel b: mean iSNVs per sample for various MAF thresholds
isnvs_path = os.path.join(
    base_processed_dir,
    '10x/total_isnvs_per_sample_and_maf_all_masks_depth_1000x/*.parquet'
)
ddf = dd.read_parquet(
    isnvs_path,
    engine='pyarrow',
    columns=['ID','Site','Mask_Condition','MAF_Threshold_Percent','iSNV_Count']
).categorize(columns=['Site','Mask_Condition'])

mean_df = (
    ddf
    .groupby(['Site','Mask_Condition','MAF_Threshold_Percent'], observed=True)
    .iSNV_Count.mean()
    .compute()
    .reset_index()
)

pivoted = mean_df.pivot_table(
    index='MAF_Threshold_Percent',
    columns=['Site','Mask_Condition'],
    values='iSNV_Count',
    observed=True
)
thresholds = sorted(pivoted.index)

# tag mask type for panel c
mask_df['Mask_Type'] = mask_df['mask'].apply(
    lambda m: 'adaptive' if m in adaptive_masks.values() else 'prevalent'
)

# reference (OXON 3% MAF, no mask)
ref_site = site_key('OXON ve-SEQ')
ref_thresh = 3
try:
    ref_y_value = pivoted[(ref_site, 'no_mask')].reindex(thresholds).loc[ref_thresh]
except Exception:
    ref_y_value = np.nan


#########################################################################

# Plotting 
plt.style.use("plot_style_settings.mplstyle")

fig = plt.figure(figsize=(7.5, 9.5), dpi = 600)
outer = GridSpec(4, 1, height_ratios=[2, 1, 0.1, 1.8], hspace=0.4)

# Panel a
ncols_a = 4
nrows_a = ceil(len(sites) / ncols_a)
gs_a = GridSpecFromSubplotSpec(
    nrows_a, ncols_a, subplot_spec=outer[0], hspace=0.3, wspace=0.1
)

curve_color = "black"
highlight_color = "crimson"

# Threshold for minimum percentage of mask files containing position
prevalence_threshold = 0.2   # 20%

first_ax = None 

for idx, site in enumerate(sites):
    ax = fig.add_subplot(gs_a[idx // ncols_a, idx % ncols_a])
    if first_ax is None:
        first_ax = ax
    x, y = site_position_freq.get(site, ([], []))

    ax.plot(x, y, linestyle='-', color=curve_color, alpha=0.3, linewidth=0.4)
    ax.scatter(x, y, s=3, color=curve_color, alpha=0.5, edgecolors='none')

    # 20% line + highlighted points
    ax.axhline(prevalence_threshold, color='black', linestyle=':', linewidth=0.5)
    for px, py in zip(x, y):
        if py > prevalence_threshold:
            ax.scatter(px, py, color=highlight_color, s=3, alpha=0.7,
                       edgecolors='none', zorder=10)

    ax.set_title(site)
    if idx == 0:
        ax.set_ylabel("Proportion of mask schemes")
    if idx // ncols_a == nrows_a - 1:
        ax.set_xlabel("Genomic Position")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True)
    ax.tick_params(axis="both", which="both", bottom=True, left=True)

    # y-label on leftmost in each row
    if idx % ncols_a == 0:
        ax.set_ylabel("Proportion of mask schemes")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
    else:
        ax.yaxis.set_tick_params(labelleft=False)

    # Legend for highlighted points
    handle_all = mlines.Line2D([], [], color=highlight_color, marker='o',
                           linestyle='None', markersize=3,
                           label=f'Positions in >{int(prevalence_threshold*100)}% of masks')
    first_ax.legend(handles=[handle_all],  bbox_to_anchor=(2.5, 1.35))


# Panel b
ax_b = fig.add_subplot(outer[1])

summary = (
    mask_df
        .rename(columns={'Centre_protocol': 'site'})
        .pivot(index='site', columns='Mask_Type', values='Masked_Count')
        .reindex(base_sites)
)

x = np.arange(len(sites))
w = 0.28
adaptive_vals = summary['adaptive'].values if 'adaptive' in summary.columns else np.zeros_like(x, dtype=float)
prevalent_vals = summary['prevalent'].values if 'prevalent' in summary.columns else np.zeros_like(x, dtype=float)

ax_b.bar(x - w/2, adaptive_vals, w, color='black',
         label='Adaptive mask (1.5% shared at x% MAF)')
ax_b.bar(x + w/2, prevalent_vals, w, color='crimson',
         label='Prevalent mask (in >20% of mask schemes)')

ax_b.set_ylim(0,200)
ax_b.set_xticks(x)
ax_b.tick_params(axis="both", which="both", bottom=True, left=True)
ax_b.set_xticklabels(sites, rotation=30, ha='right')
ax_b.tick_params(axis='x', pad=1)
ax_b.set_ylabel('Number of masked positions')
ax_b.grid(True)
ax_b.legend(frameon=False, ncols=2, loc='upper right')

# Panel c
ncols_c = 4
nrows_c = ceil(len(sites) / ncols_c)
gs_c = GridSpecFromSubplotSpec(
    nrows_c, ncols_c, subplot_spec=outer[3], hspace=0.3, wspace=0.1
)

for idx, site in enumerate(sites):
    ax = fig.add_subplot(gs_c[idx // ncols_c, idx % ncols_c])

    base = site_key(site)

    # no mask curve
    if (base, 'no_mask') in pivoted.columns:
        y_no = pivoted[(base, 'no_mask')].reindex(thresholds).values
        ax.plot(thresholds, y_no, '-o', color='darkgrey',
                linewidth=0.8, markersize=1.2)

    # adaptive (1.5% shared at X% MAF)
    adaptive = adaptive_masks.get(base)
    if adaptive and (base, adaptive) in pivoted.columns:
        y_adaptive = pivoted[(base, adaptive)].reindex(thresholds).values
        ax.plot(thresholds, y_adaptive, '-o', color='black',
        linewidth=0.8, markersize=1.2)

    # prevalent (>20% masks)
    prevalent = prevalent_masks.get(base)
    if prevalent and (base, prevalent) in pivoted.columns:
        y_prevalent = pivoted[(base, prevalent)].reindex(thresholds).values
        ax.plot(thresholds, y_prevalent, '-o', color='crimson',
                linewidth=0.8, markersize=1.2)
        # highlight MAF closest to reference level
        if np.isfinite(ref_y_value):
            diffs = np.abs(y_prevalent - ref_y_value)
            if np.isfinite(diffs).any():
                best_idx = int(np.nanargmin(diffs))
                ax.axvline(thresholds[best_idx], linestyle=':', color='darkorange', linewidth=0.9)

    # reference horizontal line
    if np.isfinite(ref_y_value):
        ax.axhline(ref_y_value, linestyle=':', color='darkgrey', linewidth=0.9)

    # styling
    row, col = divmod(idx, ncols_c)
    ax.set_title(site)
    ax.tick_params(axis="both", which="both", bottom=True, left=True)
    ax.set_xticks(all_mafs)
    ax.set_xticklabels([m if m % 2 == 0 else '' for m in all_mafs])
    ax.tick_params(axis='x')
    ax.set_ylim(0, 40)
    ax.grid(True)
    if col == 0:
        ax.set_ylabel('Mean iSNVs per sample')
    else:
        ax.yaxis.set_tick_params(labelleft=False)
    if row == nrows_c - 1:
        ax.set_xlabel('MAF Threshold (%)')

handles_c = [
    plt.Line2D([0], [0], color="darkgrey", lw=0.6, marker="o", markersize=2, label="no mask"),
    plt.Line2D([0], [0], color="black", lw=0.6, marker="o", markersize=2, label="masked - adaptive mask"),
    plt.Line2D([0], [0], color="crimson", lw=0.6, marker="o", markersize=2, label="masked - prevalent mask"),
    plt.Line2D([0], [0], color="darkorange", lw=0.9, linestyle=":",
           label="analysis MAF (masked ≈10 iSNVs)"),
]

fig.legend(
    handles=handles_c,
    loc="upper right",
    ncols=5,
    bbox_to_anchor=(0.90, 0.38),
    bbox_transform=fig.transFigure,
    frameon=False
)

# Panel labels
fig.text(0.06, 0.91, 'a', fontweight='bold', fontsize=8)
fig.text(0.06, 0.60, 'b', fontweight='bold', fontsize=8)
fig.text(0.06, 0.38, 'c', fontweight='bold', fontsize=8)

plot_filename = 'figureS6_prevalent_mask_strategy.png'
plot_path = os.path.join(figures_dir, plot_filename)
plt.savefig(plot_path, format='png', bbox_inches='tight')
plt.close()

print(f"Figure saved as '{plot_path}'")
