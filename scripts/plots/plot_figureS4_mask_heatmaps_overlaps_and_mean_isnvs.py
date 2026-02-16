import os
import re
import math
import json
import itertools

import numpy as np
import pandas as pd
import dask.dataframe as dd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator, FuncFormatter, ScalarFormatter
from matplotlib.lines import Line2D


base_processed_dir = "../../processed_data"
figures_dir = "../../figures/supplemental"

cov_folder = "10x"
mask_folder = "mask_files_depth_1000x"

site_to_folder_map = {
    "NORT ARTIC 3": "NORT_ARTIC_3_ILLUMINA",
    "NORT ARTIC 4": "NORT_ARTIC_4_ILLUMINA",
    "NORT ARTIC 4.1": "NORT_ARTIC_4.1_ILLUMINA",
    "NORW ARTIC ?": "NORW_ARTIC_4.1_ILLUMINA",
    "OXON ve-SEQ": "OXON_VeSeq_ILLUMINA",
    "PHEC ARTIC 3": "PHEC_ARTIC_3_ILLUMINA",
    "SANG ARTIC 3": "SANG_ARTIC_3_ILLUMINA",
    "SANG ARTIC 4.1": "SANG_ARTIC_4.1_ILLUMINA"
}
sites = list(site_to_folder_map.keys())

site_key_from_pretty = {
    "NORT ARTIC 3": "NORT_ARTIC_3",
    "NORT ARTIC 4": "NORT_ARTIC_4",
    "NORT ARTIC 4.1": "NORT_ARTIC_4.1",
    "NORW ARTIC ?": "NORW_ARTIC_4.1",
    "OXON ve-SEQ": "OXON_VeSeq",
    "PHEC ARTIC 3": "PHEC_ARTIC_3",
    "SANG ARTIC 3": "SANG_ARTIC_3",
    "SANG ARTIC 4.1": "SANG_ARTIC_4.1",
}


# Panel a: Heatmap of masked positions
# Regex pattern to match filenames like "1.5%shared_3%MAF.csv"
pattern = re.compile(r'(\d+(?:\.\d+)?)%shared_(\d+)%MAF\.csv')

def load_masking_linecount(folder, shared_percentage, maf_percentage):
    base_dir = os.path.join(base_processed_dir, cov_folder, folder, mask_folder)
    file_name = f"{shared_percentage}%shared_{maf_percentage}%MAF.csv"
    file_path = os.path.join(base_dir, file_name)
    if not os.path.exists(file_path):
        return 0
    try:
        with open(file_path, "r") as f:
            return sum(1 for _ in f)
    except Exception:
        return 0

# Collect heatmap records
heat_records = []
for site in sites:
    folder = site_to_folder_map.get(site)
    if not folder:
        continue
    folder_path = os.path.join(base_processed_dir, cov_folder, folder, mask_folder)
    if not os.path.isdir(folder_path):
        continue
    for fname in os.listdir(folder_path):
        m = pattern.match(fname)
        if not m:
            continue
        shared_str, maf_str = m.groups()
        shared_pct = float(shared_str)
        maf_pct    = int(maf_str)
        num_masked = load_masking_linecount(folder, shared_pct, maf_pct)
        heat_records.append({
            "site": site,
            "shared_pct": shared_pct,
            "maf_pct": maf_pct,
            "num_masked_pos": num_masked
        })

heat_df = pd.DataFrame(heat_records)
if heat_df.empty:
    heat_df = pd.DataFrame(columns=["site","shared_pct","maf_pct","num_masked_pos"])

# Global color scale (log) for panel a
if (heat_df["num_masked_pos"] > 0).any():
    global_vmin = int(heat_df.loc[heat_df["num_masked_pos"] > 0, "num_masked_pos"].min())
    global_vmax = int(heat_df["num_masked_pos"].max())
    global_vmin = max(global_vmin, 1)
    global_vmax = max(global_vmax, 1)
else:
    global_vmin, global_vmax = 1, 1
heat_norm = LogNorm(vmin=global_vmin, vmax=global_vmax, clip=True)

# Determine consistent axes for all heatmaps
full_maf = sorted(heat_df["maf_pct"].unique()) if not heat_df.empty else []
full_shared = sorted(heat_df["shared_pct"].unique()) if not heat_df.empty else []


# panel b: scatter of mean iSNVs per sample for each masking condition and analysis MAF threshold
isnvs_path = os.path.join(base_processed_dir,cov_folder,
    "total_isnvs_per_sample_and_maf_all_masks_depth_1000x", "*.parquet"
)
ddf = dd.read_parquet(
    isnvs_path,
    engine="pyarrow",
    columns=["ID", "Site", "Mask_Condition", "MAF_Threshold_Percent", "iSNV_Count"]
).categorize(columns=["Site", "Mask_Condition"])

mean_df = (
    ddf.groupby(["Site", "Mask_Condition", "MAF_Threshold_Percent"], observed=True)
       .iSNV_Count.mean()
       .compute()
       .reset_index()
)

pivoted = mean_df.pivot_table(
    index="MAF_Threshold_Percent",
    columns=["Site", "Mask_Condition"],
    values="iSNV_Count",
    observed=True
)

thresholds = sorted(pivoted.index)

pattern = re.compile(r"(?P<shared>[0-9]*\.?[0-9]+)%shared_(?P<maskmaf>[0-9]+)%MAF")

rows = []
for pretty_site in sites:
    site_key = site_key_from_pretty.get(pretty_site)
    if site_key is None:
        continue

    site_cols = [c for c in pivoted.columns if c[0] == site_key]
    for _, mask_name in site_cols:
        if mask_name == "no_mask":
            continue
        m = pattern.match(mask_name)
        if not m:
            continue

        shared = float(m.group("shared"))
        maskmaf = int(m.group("maskmaf"))
        y = pivoted[(site_key, mask_name)].reindex(thresholds).values

        for analysis_maf, mean_isnvs in zip(thresholds, y):
            if not np.isfinite(mean_isnvs):
                continue
            rows.append({
                "site_pretty": pretty_site,
                "analysis_maf": int(analysis_maf),
                "mask_maf": int(maskmaf),
                "shared_pct": float(shared),
                "mean_isnvs": float(mean_isnvs),
            })

df_long = pd.DataFrame(rows)


# Panel c: mask overlap analysis
filtered_shared = [x/2 for x in range(1, 41)]
filtered_maf = list(range(2, 21))

all_site_data = []
all_stack_counts = []

for site in sites:
    csv_path = os.path.join(
        base_processed_dir, cov_folder,
        site_to_folder_map[site],
        mask_folder, "all_masks.csv"
    )
    if not os.path.isfile(csv_path):
        all_site_data.append((site, None))
        continue

    df = pd.read_csv(csv_path)
    df['positions'] = df['positions'].apply(lambda x: set(map(int, json.loads(x))))
    df['shared'] = df['mask_file'].str.extract(r'([0-9\.]+)%shared')[0].astype(float)
    df['maf'] = df['mask_file'].str.extract(r'([0-9\.]+)%MAF')[0].astype(float)
    df = df[df['shared'].isin(filtered_shared) & df['maf'].isin(filtered_maf)].reset_index(drop=True)
    if len(df) < 2:
        all_site_data.append((site, None))
        continue

    recs = []
    for i, j in itertools.combinations(df.index, 2):
        if df.at[i,'shared'] == df.at[j,'shared'] or df.at[i,'maf'] == df.at[j,'maf']:
            continue
        A, B = df.at[i,'positions'], df.at[j,'positions']
        small = min(len(A), len(B)) if min(len(A), len(B)) > 0 else 1
        ov = len(A & B) / small
        diff = abs(len(A) - len(B))
        recs.append({'small_size': small, 'overlap': ov, 'diff_size': diff})

    comp = pd.DataFrame(recs)
    if comp.empty:
        all_site_data.append((site, None))
        continue

    comp['stack_count'] = comp.groupby(['small_size','overlap'])['diff_size'].transform('size')
    all_stack_counts.extend(comp['stack_count'])
    all_site_data.append((site, comp))

if all_stack_counts:
    sc_min = max(min(all_stack_counts), 1)
    sc_max = max(all_stack_counts)
else:
    sc_min, sc_max = 1, 1
overlap_norm = LogNorm(vmin=sc_min, vmax=sc_max, clip=True)


########################################################
## Plotting
########################################################
plt.style.use("plot_style_settings.mplstyle")

n_cols = 4
n_sites = len(sites)
rows = math.ceil(n_sites / n_cols)

fig = plt.figure(figsize=(7.5, 10.5))
gs = fig.add_gridspec(
    nrows=3, ncols=2,
    width_ratios=[20, 1.3], height_ratios=[1, 1, 1],
    left=0.06, right=0.98, bottom=0.06, top=0.98, wspace=0.2, hspace=0.22
)
# Color maps
rocket_cmap = sns.color_palette("rocket", as_cmap=True)
magma_cmap = plt.get_cmap("magma_r")

# Panel a
subgs_a = gs[0, 0].subgridspec(rows, n_cols, wspace=0.2, hspace=0.28)
axes_a = []
for idx in range(rows * n_cols):
    if idx < n_sites:
        ax = fig.add_subplot(subgs_a[idx // n_cols, idx % n_cols])
    else:
        ax = fig.add_subplot(subgs_a[idx // n_cols, idx % n_cols])
        ax.set_visible(False)
    axes_a.append(ax)

# Draw heatmaps
for idx, site in enumerate(sites):
    ax = axes_a[idx]
    site_df = heat_df[heat_df["site"] == site].copy()
    if site_df.empty or len(full_maf) == 0 or len(full_shared) == 0:
        ax.set_axis_off()
        continue

    pivot = site_df.pivot(index="shared_pct", columns="maf_pct", values="num_masked_pos")
    pivot = pivot.reindex(index=full_shared, columns=full_maf)

    sns.heatmap(
        pivot,
        ax=ax,
        cmap=magma_cmap,
        norm=heat_norm,
        cbar=False,
        annot=False,
        square=False
    )

    # axis cosmetics + labels on outer edges
    row, col = divmod(idx, n_cols)
    ax.set_title(site)
    ax.invert_yaxis()

    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.tick_params(top=False, right=False)

    x_all = np.arange(len(pivot.columns))+0.5
    y_all = np.arange(len(pivot.index))+0.5
    x_top = (np.arange(len(pivot.columns)) % 2 == 0)
    y_top = (np.arange(len(pivot.index)) % 3 == 0)
    x_major = x_all[x_top]
    x_minor = x_all[~x_top]
    y_major = y_all[y_top]
    y_minor = y_all[~y_top]

    ax.set_xticks(x_major)
    ax.set_xticklabels([str(c) for c in np.array(pivot.columns)[x_top]])
    ax.set_xticks(x_minor, minor=True)
    ax.tick_params(axis="x", pad=1)

    ax.set_yticks(y_major)
    ax.set_yticklabels([str(v) for v in np.array(pivot.index)[y_top]],
                   rotation=0)
    ax.set_yticks(y_minor, minor=True)

    if row == rows - 1:
        ax.set_xlabel("MAF threshold (%)")
    else:
        ax.set_xlabel("")
    if col == 0:
        ax.set_ylabel("% Shared threshold")
    else:
        ax.set_ylabel("")


# Colorbar on the right of panel a
sm_a = plt.cm.ScalarMappable(cmap=magma_cmap, norm=heat_norm)
sm_a.set_array([])
cax_a = fig.add_axes([0.89, 0.70, 0.015, 0.28])
cbar_a = fig.colorbar(sm_a, cax=cax_a)
cbar_a.set_label("Number of masked positions")

locator = LogLocator(base=10.0, subs=np.arange(1, 6.1))
formatter = FuncFormatter(lambda x, pos: f"{int(x)}")
cbar_a.ax.yaxis.set_major_locator(locator)
cbar_a.ax.yaxis.set_major_formatter(formatter)


# panel b
subgs_b = gs[1, 0].subgridspec(rows, n_cols, wspace=0.2, hspace=0.28)

# colour = mask MAF
rocket_cmap = sns.color_palette("rocket", as_cmap=True)
maf_vmin = float(df_long["mask_maf"].min())
maf_vmax = float(df_long["mask_maf"].max())
maf_norm = mpl.colors.Normalize(vmin=maf_vmin, vmax=maf_vmax)
sm_b = mpl.cm.ScalarMappable(cmap=rocket_cmap, norm=maf_norm)
sm_b.set_array([])

# size = shared %
shared_min = float(df_long["shared_pct"].min())
shared_max = float(df_long["shared_pct"].max())
size_min = 1
size_max = 20

def shared_to_s(vals):
    vals = np.asarray(vals, dtype=float)
    if shared_max <= shared_min:
        return np.full_like(vals, (size_min + size_max) / 2.0, dtype=float)
    return size_min + (vals - shared_min) / (shared_max - shared_min) * (size_max - size_min)

# broken y settings
y_low_max = 40
y_high_min = 41
y_high_max = float(df_long["mean_isnvs"].max()) + 50

rng = np.random.default_rng(0)
jitter = 0.10
alpha = 0.30

axes_b_top = []
axes_b_bot = []

for idx, site in enumerate(sites):
    row, col = divmod(idx, n_cols)

    inner = subgs_b[row, col].subgridspec(2, 1, height_ratios=[0.25, 1.0], hspace=0.045)
    ax_top = fig.add_subplot(inner[0, 0])
    ax_bot = fig.add_subplot(inner[1, 0], sharex=ax_top)

    axes_b_top.append(ax_top)
    axes_b_bot.append(ax_bot)

    d = df_long[df_long["site_pretty"] == site].copy()
    if d.empty:
        ax_top.axis("off")
        ax_bot.axis("off")
        continue

    x = d["analysis_maf"].to_numpy(float)
    y = d["mean_isnvs"].to_numpy(float)
    maf = d["mask_maf"].to_numpy(float)
    shared = d["shared_pct"].to_numpy(float)

    xj = x + rng.uniform(-jitter, jitter, size=len(x))
    sizes = shared_to_s(shared)

    low = y <= y_low_max
    high = y >= y_high_min

    if high.any():
        ax_top.scatter(xj[high], y[high], c=maf[high], s=sizes[high],
                       cmap=rocket_cmap, norm=maf_norm, alpha=alpha, linewidths=0)
    if low.any():
        ax_bot.scatter(xj[low], y[low], c=maf[low], s=sizes[low],
                       cmap=rocket_cmap, norm=maf_norm, alpha=alpha, linewidths=0)

    ax_bot.set_ylim(-1, y_low_max)
    ax_top.set_ylim(y_high_min, y_high_max)

    # break styling
    ax_top.spines["bottom"].set_linestyle("--")
    ax_top.spines["bottom"].set_color("grey")
    ax_top.spines["bottom"].set_linewidth(0.3)
    ax_bot.spines["top"].set_linestyle("--")
    ax_bot.spines["top"].set_color("grey")
    ax_bot.spines["top"].set_linewidth(0.3)

    # ticks and grids
    ax_top.tick_params(axis="both", bottom=False, left=True, labelbottom=False)
    ax_bot.tick_params(axis="both", bottom=True, left=True)

    ax_bot.set_xticks(sorted(set(thresholds)))
    ax_bot.set_xticklabels([t if (t % 2 == 0) else "" for t in sorted(set(thresholds))])
    ax_bot.set_yticks(np.arange(0, y_low_max+1, 10))

    ax_top.grid(True, axis="y", linewidth=0.3, alpha=0.3)
    ax_bot.grid(True, axis="y", linewidth=0.3, alpha=0.3)

    ax_top.set_title(site)

    if col == 0:
        ax_bot.set_ylabel("Mean iSNVs per sample")

    if row == rows - 1:
        ax_bot.set_xlabel("Analysis MAF threshold (%)")
    else:
        ax_bot.set_xlabel("")

# panel b legend for shared % size scale
sax = fig.add_axes([0.87, 0.58, 0.06, 0.06])
sax.set_xlim(0, 1)
sax.set_ylim(0, 1)
sax.axis("off")
sax.text(0.7, 1.02, "Shared (%)", ha="center", va="bottom", fontsize=7)

# panel b legend for MAF colour scale
cax_b = fig.add_axes([0.89, 0.37, 0.015, 0.20])
cbar_b = fig.colorbar(sm_b, cax=cax_b)
cbar_b.set_label("Mask MAF (%)")


shared_ticks = [0.5, 2, 5, 10, 15, 20]
xs = np.full(len(shared_ticks), 0.5)
ys = np.linspace(0.10, 0.90, len(shared_ticks))
sax.scatter(xs, ys, s=shared_to_s(shared_ticks), color="dimgray", alpha=0.6, linewidths=0)
for y_pos, sv in zip(ys, shared_ticks):
    sax.text(0.85, y_pos, f"{sv:g}", ha="left", va="center", fontsize=6)


# Panel c
subgs_c = gs[2, 0].subgridspec(rows, n_cols, wspace=0.2, hspace=0.28)
axes_c = []
for idx in range(rows * n_cols):
    if idx < n_sites:
        ax = fig.add_subplot(subgs_c[idx // n_cols, idx % n_cols])
    else:
        ax = fig.add_subplot(subgs_c[idx // n_cols, idx % n_cols])
        ax.set_visible(False)
    axes_c.append(ax)

last_sc = None
size_scale = 0.1  # marker area scale

for idx, (site, comp) in enumerate(all_site_data):
    ax = axes_c[idx]
    ax.set_title(site)
    if comp is None or comp.empty:
        ax.axis('off')
        continue

    comp = comp.copy()
    comp['sz'] = comp['diff_size'].astype(float) * size_scale

    sc = ax.scatter(
        comp['small_size'], comp['overlap'],
        s=comp['sz'],
        c=comp['stack_count'],
        cmap=rocket_cmap,
        norm=overlap_norm,
        linewidth=0.3, alpha=0.7
    )
    last_sc = sc
    row_c, col_c = divmod(idx, n_cols)
    if row_c == rows - 1:
        ax.set_xlabel("Mask size (smallest mask)")
    else:
        ax.set_xlabel("")

    if col_c == 0:
        ax.set_ylabel("Overlap coefficient")
    else:
        ax.set_ylabel("")
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 1.0)

    ax.tick_params(axis="both", which="both", bottom=True, left=True)

# Panel c colorbar
cax_c = fig.add_axes([0.89, 0.04, 0.015, 0.20])
if last_sc is not None:
    sm_c = plt.cm.ScalarMappable(cmap=rocket_cmap, norm=overlap_norm)
    sm_c.set_array([])
    cbar_c = fig.colorbar(sm_c, cax=cax_c)
    cbar_c.set_label("Number of comparisons with same size/overlap")
    fmt = ScalarFormatter()
    fmt.set_scientific(False); fmt.set_useOffset(False)
    cbar_c.ax.yaxis.set_major_formatter(fmt)
else:
    cax_c.axis("off")

# Size legend for diff_size
legend_values = [1, 10, 100, 1000]
handles, labels = [], []
for val in legend_values:
    area = val * size_scale
    diameter = math.sqrt(area)
    h = Line2D(
        [], [], linestyle="",
        marker="o",
        markersize=diameter,
        markerfacecolor="black",
        markeredgecolor="black",
        alpha=0.7,
        linewidth=0.05
    )
    handles.append(h)
    labels.append(f"{val} positions")

leg = fig.legend(
    handles, labels,
    title="Mask-size difference",
    loc="upper right",
    bbox_to_anchor=(0.99, 0.34),
    frameon=False, labelspacing=1, handletextpad=1, title_fontsize=7
)
# Align legend text to the right
try:
    leg._legend_box.align = 'right'
except Exception:
    pass

# panel labels
fig.text(0.01, 0.99, 'a', fontsize = 8, fontweight='bold')
fig.text(0.01, 0.66, 'b', fontsize = 8, fontweight='bold')
fig.text(0.01, 0.33, 'c', fontsize = 8, fontweight='bold')

outfile = os.path.join(figures_dir, "figureS4_mask_heatmaps_overlaps_and_mean_isnvs.png")
plt.savefig(outfile, dpi=600, bbox_inches="tight")
plt.close()
print(f"Figure saved as '{outfile}'")