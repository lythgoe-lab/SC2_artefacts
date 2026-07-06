import os
import re
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.colors import to_rgb
import seaborn as sns


base_dir = "../../"
base_processed_dir = "../../processed_data/"
figures_dir = "../../figures/"

depth_threshold = 1000
cov_folder = '10x'

maf_thresholds = [3, 5, 10, 20]
maf_scatter = 3

site_to_folder_map = {
    "OXON VeSeq": "OXON_VeSeq_ILLUMINA",
    "NORT ARTIC 3": "NORT_ARTIC_3_ILLUMINA",
    "NORT ARTIC 4": "NORT_ARTIC_4_ILLUMINA",
    "NORT ARTIC 4.1": "NORT_ARTIC_4.1_ILLUMINA",
    "NORW ARTIC ?": "NORW_ARTIC_4.1_ILLUMINA",
    "PHEC ARTIC 3": "PHEC_ARTIC_3_ILLUMINA",
    "SANG ARTIC 3": "SANG_ARTIC_3_ILLUMINA",
    "SANG ARTIC 4.1": "SANG_ARTIC_4.1_ILLUMINA"
}
subset_sites = [
    "NORT ARTIC 3", "NORT ARTIC 4",
    "NORT ARTIC 4.1", "NORW ARTIC ?",
    "PHEC ARTIC 3", "SANG ARTIC 3", "SANG ARTIC 4.1"
]

sample_summary_csv = os.path.join(base_dir, 'processed_data', 'samples_per_site_cov_summary.csv')

summary_df = pd.read_csv(sample_summary_csv, low_memory=False)
summary_df = summary_df[summary_df['Cov_Threshold'] == cov_folder]

# Function to load long-format iSNV data for one site
def load_long_isnv(site_name: str, maf_depth_pattern, base_dir, cov_folder, depth_threshold=None):
    lab_folder = site_to_folder_map[site_name]
    file_path = os.path.join(base_dir, "processed_data", cov_folder, lab_folder, 'samples_per_iSNV_position.csv')
    if not os.path.exists(file_path):
        print(f"[Warning] File not found: {file_path}. Skipping {site_name}.")
        return None

    data = pd.read_csv(file_path, low_memory=False)
    data["site"] = site_name

    non_threshold_cols = ["Position", "site"]
    threshold_cols = [c for c in data.columns if c not in non_threshold_cols]

    long_data = data.melt(
        id_vars=non_threshold_cols,
        value_vars=threshold_cols,
        var_name="Threshold_Column",
        value_name="Num_Samples"
    )
    parsed = long_data["Threshold_Column"].str.extract(maf_depth_pattern).astype(int)
    long_data["MAF"] = parsed["MAF"]
    long_data["depth"] = parsed["depth"]

    if depth_threshold is not None:
        long_data = long_data[long_data["depth"] == depth_threshold]

    return long_data

# prepare iSNV data for all plots
pattern = re.compile(r"Samples_MAF_(?P<MAF>\d+)%_depth_(?P<depth>\d+)")

site_total_samples = {}
for site in subset_sites:
    folder_name = site_to_folder_map[site]
    row = summary_df[summary_df['Site'] == folder_name]
    if row.empty:
        raise KeyError(f"No summary row found for site {site} (folder {folder_name})")
    site_total_samples[site] = int(row.iloc[0]['Total_Samples'])

# prepare dataframe of iSNVs per position for each site
all_long_data_site = []

for site_name in subset_sites:
    long_data = load_long_isnv(site_name, pattern, base_dir, cov_folder, depth_threshold)
    if long_data is None:
        continue

    sub = long_data[
        (long_data["MAF"].isin(maf_thresholds)) &
        (long_data["Num_Samples"] > 0)
    ].copy()

    total_samples = site_total_samples.get(site_name)
    sub['Proportion'] = sub['Num_Samples'] / total_samples
    all_long_data_site.append(sub)

all_long_data = pd.concat(all_long_data_site, ignore_index=True)

# prepare combined data with scheme labels
ARTIC_4_1_sites = ["SANG ARTIC 4.1", "NORT ARTIC 4.1", "NORW ARTIC ?"]
ARTIC_3_sites = ["PHEC ARTIC 3", "SANG ARTIC 3", "NORT ARTIC 3"]
ARTIC_4_sites = ["NORT ARTIC 4"]

site_to_scheme = {}
for s in ARTIC_4_1_sites:
    site_to_scheme[s] = "ARTIC 4.1"
for s in ARTIC_3_sites:
    site_to_scheme[s] = "ARTIC 3"
for s in ARTIC_4_sites:
    site_to_scheme[s] = "ARTIC 4"

all_long_data["scheme"] = all_long_data["site"].map(site_to_scheme)

# get ARTIC primer and amplicon data
# Load primer positions
PRIMER_BED_ARTIC_3 = os.path.join(base_dir, "ARTIC_3.primer.bed")
PRIMER_BED_ARTIC_4 = os.path.join(base_dir, "ARTIC_4.primer.bed")
PRIMER_BED_ARTIC_4_1 = os.path.join(base_dir, "ARTIC_4.1.primer.bed")

def load_amplicons_from_bed(bed_path: str, scheme_name: str):
    """
    Load primer BED and return:
      - amplicon summary (one row per amplicon, only using primers that are not "alt")
      - primer table (all, including alt variants)
    """
    cols = ["chrom", "start", "end", "primer_name", "pool", "strand", "sequence"]
    df = pd.read_csv(bed_path, sep="\t", header=None, names=cols)

    # classify side and amplicon ID
    def extract_amplicon(name) -> int:
        s = str(name)
        m = re.search(r"_(\d+)_", s)
        if m is not None:
            return int(m.group(1))
        if s.isdigit():
            return int(s)
        raise ValueError(f"Could not parse amplicon number from primer name: {name}")

    df["amplicon"] = df["primer_name"].apply(extract_amplicon)
    df["side"] = df["primer_name"].apply(
        lambda x: "LEFT" if "LEFT" in str(x).upper() else "RIGHT"
    )

    # primers table: all primers (including alt)
    primers = df[[
        "chrom", "amplicon", "side", "start", "end"
    ]].copy()
    primers["scheme"] = scheme_name

    # amplicon summary from non-alt primers only
    df_non_alt = df[~df["primer_name"].str.contains("alt", case=False, na=False)].copy()

    left = (
        df_non_alt[df_non_alt["side"] == "LEFT"]
        .groupby(["chrom", "amplicon"], as_index=False)
        .agg(
            left_start=("start", "min"),
            left_end=("end", "max"),
        )
    )

    right = (
        df_non_alt[df_non_alt["side"] == "RIGHT"]
        .groupby(["chrom", "amplicon"], as_index=False)
        .agg(
            right_start=("start", "min"),
            right_end=("end", "max"),
        )
    )

    amp = pd.merge(left, right, on=["chrom", "amplicon"], how="inner")
    amp["amplicon_start"] = amp["left_start"]
    amp["amplicon_end"] = amp["right_end"]
    amp["amplicon_length"] = amp["amplicon_end"] - amp["amplicon_start"]
    amp["scheme"] = scheme_name

    amplicons = amp[[
        "scheme", "amplicon", "chrom",
        "left_start", "left_end",
        "right_start", "right_end",
        "amplicon_start", "amplicon_end", "amplicon_length"
    ]].copy()

    return amplicons, primers


def compute_amplicon_distances(all_amplicons: pd.DataFrame) -> pd.DataFrame:
    """Compute distances from each position to amplicon ends and primers for that amplicon"""

    rows = []

    for _, row in all_amplicons.iterrows():
        scheme = row["scheme"]
        amp_id = row["amplicon"]

        start0 = int(row["amplicon_start"])
        end0 = int(row["amplicon_end"])
        left_start0 = int(row["left_start"])
        left_end0 = int(row["left_end"])
        right_start0 = int(row["right_start"])
        right_end0 = int(row["right_end"])

        pos_start = start0 + 1
        pos_end = end0
        if pos_end < pos_start:
            continue

        positions = np.arange(pos_start, pos_end + 1)

        dist_left_end = positions - pos_start
        dist_right_end = pos_end - positions
        dist_within_amplicon = np.minimum(dist_left_end, dist_right_end)

        left_start1 = left_start0 + 1
        left_end1 = left_end0
        right_start1 = right_start0 + 1
        right_end1 = right_end0

        left_inside = (positions >= left_start1) & (positions <= left_end1)
        dist_to_left = np.empty_like(positions, dtype=np.int64)
        dist_to_left[left_inside] = 0
        if (~left_inside).any():
            dl = np.abs(positions[~left_inside] - left_start1)
            dr = np.abs(positions[~left_inside] - left_end1)
            dist_to_left[~left_inside] = np.minimum(dl, dr)

        right_inside = (positions >= right_start1) & (positions <= right_end1)
        dist_to_right = np.empty_like(positions, dtype=np.int64)
        dist_to_right[right_inside] = 0
        if (~right_inside).any():
            dl = np.abs(positions[~right_inside] - right_start1)
            dr = np.abs(positions[~right_inside] - right_end1)
            dist_to_right[~right_inside] = np.minimum(dl, dr)

        tmp = pd.DataFrame({
            "scheme": scheme,
            "amplicon": amp_id,
            "position": positions,
            "dist_within_amplicon": dist_within_amplicon,
            "dist_to_left_covering_primer": dist_to_left,
            "dist_to_right_covering_primer": dist_to_right,
        })
        rows.append(tmp)

    per_amp_pos = pd.concat(rows, ignore_index=True)

    per_pos_summary = (
        per_amp_pos
        .groupby(["scheme", "position"], as_index=False)
        .agg(
            min_dist_within_amplicon=("dist_within_amplicon", "min"),
            min_covering_left_primer_dist=("dist_to_left_covering_primer", "min"),
            min_covering_right_primer_dist=("dist_to_right_covering_primer", "min"),
        )
    )
    return per_pos_summary

# build combined tables
artic3_amp, artic3_prim = load_amplicons_from_bed(PRIMER_BED_ARTIC_3, "ARTIC 3")
artic4_amp, artic4_prim = load_amplicons_from_bed(PRIMER_BED_ARTIC_4, "ARTIC 4")
artic4_1_amp, artic4_1_prim = load_amplicons_from_bed(PRIMER_BED_ARTIC_4_1, "ARTIC 4.1")

all_amplicons = pd.concat([artic3_amp, artic4_amp, artic4_1_amp], ignore_index=True)
all_primers = pd.concat([artic3_prim, artic4_prim, artic4_1_prim], ignore_index=True)

amplicon_dist_all = compute_amplicon_distances(all_amplicons)

# Merge amplicon distances with iSNV data
merged_isnv_dists = all_long_data.merge(
    amplicon_dist_all,
    left_on=["Position", "scheme"],
    right_on=["position", "scheme"],
    how="left"
).drop(columns=["position"])

##############################################
# prepare for plotting

# get largest distances for plotting limits
max_dist_amp = int(merged_isnv_dists["min_dist_within_amplicon"].max())

merged_isnv_dists["min_primer_dist"] = merged_isnv_dists[
    ["min_covering_left_primer_dist", "min_covering_right_primer_dist"]
].min(axis=1)
max_dist_prim = int(merged_isnv_dists["min_primer_dist"].max())


# get original site group colors for each group from figure 1
old_group_order = ["SANG", "NORW", "NORT", "PHEC", "OXON"]
old_palette = sns.color_palette("tab10", n_colors=len(old_group_order))
old_group_colors = dict(zip(old_group_order, old_palette))

# Functions to tint and shade colors
def tint(color, amount=0.3):
    # blend toward white; amount in [0,1]
    r, g, b = to_rgb(color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)

def shade(color, amount=0.3):
    # blend toward black; amount in [0,1]
    r, g, b = to_rgb(color)
    return (r * (1 - amount), g * (1 - amount), b * (1 - amount))

# Base hues from Figure 1
nort_base = old_group_colors["NORT"]
sang_base = old_group_colors["SANG"]
norw_col = old_group_colors["NORW"]
phec_col = old_group_colors["PHEC"]
oxon_col = old_group_colors["OXON"]

# Assign colors to each site
site_colors = {
    # keep identical to Figure 1
    "NORW ARTIC ?": norw_col,
    "PHEC ARTIC 3": phec_col,

    # NORT: same tone, 3 distinguishable shades
    "NORT ARTIC 3": shade(nort_base, 0.4),
    "NORT ARTIC 4": nort_base,
    "NORT ARTIC 4.1": tint(nort_base, 0.4),

    # SANG: same tone, 2 distinguishable shades
    "SANG ARTIC 3": shade(sang_base, 0.3),
    "SANG ARTIC 4.1": tint(sang_base, 0.3),
}

# Safety checks
missing = [s for s in subset_sites if s not in site_colors]
if missing:
    raise KeyError(f"Missing colours for sites: {missing}")

# function to get z-order for each site based on original group order in figure 1
def site_zorder(site: str) -> int:
    return old_group_order.index(site.split()[0])

# colors for scheme lines
scheme_order = ["ARTIC 3", "ARTIC 4", "ARTIC 4.1"]
scheme_colors = {
    'ARTIC 3': 'royalblue',
    'ARTIC 4': 'teal',
    'ARTIC 4.1': 'forestgreen'
}
# colors for MAF thresholds
maf_color_palette = sns.color_palette('rocket', n_colors=len(maf_thresholds))
maf_to_color = {maf: maf_color_palette[i] for i, maf in enumerate(maf_thresholds)}

# Split genome in half for plotting panel a
max_pos_scatter = all_long_data["Position"].max()
max_pos_amp = all_amplicons["amplicon_end"].max()
max_pos = max(max_pos_scatter, max_pos_amp)
mid_pos = max_pos / 2.0

# sorted lists for plotting panels b and c
sites = sorted(merged_isnv_dists["site"].unique())
maf_values = sorted(merged_isnv_dists["MAF"].unique())

# binning for histograms in panel c
bin_width = 10


# Helper function to plot one scatter panel for a given x-range for panel a
def plot_scatter_panel(ax, x_min, x_max, x_label=False):
    for site in subset_sites:
        sub = all_long_data.query("site == @site and Position >= @x_min and Position <= @x_max").copy()
        if sub.empty:
            continue
        sub = sub.sort_values("Position")
        ax.scatter(
            sub["Position"],
            sub["Proportion"],
            s=1.5,
            marker="o",
            edgecolors="none",
            color=site_colors[site],
            alpha=0.7,
            label=site,
            zorder=site_zorder(site)
        )

    ax.set_ylim(0, 1)
    ax.set_xlim(x_min, x_max)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylabel("Proportion of samples\nwith iSNV")
    ax.minorticks_on()
    ax.grid(which="major", axis="x", alpha=0.3, linewidth=0.5)
    ax.grid(which="minor", axis="x", alpha=0.3, linewidth=0.2)
    ax.tick_params(axis="x", which="minor", length=0.8)
    ax.tick_params(axis="y", which="minor", length=0.8)

    if x_label:
        ax.set_xlabel("Genome position")
    else:
        ax.tick_params(labelbottom=False)

# Helper to plot one amplicon panel for a given x-range for panel a
def plot_amplicon_panel(ax, x_min, x_max):
    primer_offset = 0.03  # vertical offset from the amplicon line

    for row_idx, scheme in enumerate(scheme_order):
        amps_scheme = all_amplicons[all_amplicons["scheme"] == scheme].sort_values("amplicon")
        if amps_scheme.empty:
            continue

        # primers for this scheme
        prim_scheme = all_primers[all_primers["scheme"] == scheme]

        # keep only amplicons that intersect the x-range
        mask = (amps_scheme["amplicon_end"] >= x_min) & (amps_scheme["amplicon_start"] <= x_max)
        amps_scheme = amps_scheme[mask]
        if amps_scheme.empty:
            continue

        n_amp = len(amps_scheme)
        base_pattern = np.linspace(0, 0.1, min(2, n_amp))
        jitters = np.resize(base_pattern, n_amp)

        for j, (_, amp_row) in enumerate(amps_scheme.iterrows()):
            y = row_idx + jitters[j]

            # main amplicon span (non-alt union)
            amp_start = max(amp_row["amplicon_start"], x_min)
            amp_end = min(amp_row["amplicon_end"], x_max)
            ax.hlines(
                y,
                amp_start,
                amp_end,
                color=scheme_colors[scheme],
                linewidth=0.6,
                alpha=0.9,
                zorder=1
            )

            # all primers for this amplicon
            prim_sub = prim_scheme[prim_scheme["amplicon"] == amp_row["amplicon"]]

            for _, p in prim_sub.iterrows():
                p_start = max(p["start"], x_min)
                p_end = min(p["end"], x_max)
                if p_end <= p_start:
                    continue

                # small vertical jitter for each primer
                primer_jit = np.random.uniform(-0.015, 0.015)

                # left primers slightly above, right below, both jittered
                if p["side"] == "LEFT":
                    y_p = y + primer_offset + primer_jit
                else:  # right
                    y_p = y - primer_offset + primer_jit

                # Primer segment
                ax.hlines(
                    y_p,
                    p_start,
                    p_end,
                    color="black",
                    linewidth=1.0,
                    alpha=0.9,
                    zorder=2
                )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.2, len(scheme_order) - 0.6)
    ax.set_yticks(range(len(scheme_order)))
    ax.set_yticklabels(scheme_order)
    ax.grid(which="major", axis="x", alpha=0.3, linewidth=0.5)
    ax.grid(which="minor", axis="x", alpha=0.3, linewidth=0.2)
    ax.tick_params(axis="x", which="minor", length=0.8)



### Plotting ###

plt.style.use("plot_style_settings.mplstyle")

np.random.seed(42)

# Create figure and gridspec
fig = plt.figure(figsize=(7, 10), dpi=600)
outer = GridSpec(3, 1,
                 height_ratios=[0.5, 0.3, 0.3],
                 hspace=0.27)

# Panel a: grouped scatter plot
gs_a = GridSpecFromSubplotSpec(5, 1, subplot_spec=outer[0],
                                height_ratios=[0.4, 0.25, 0.08, 0.4, 0.25],
                                hspace=0.12)

# Top half: scatter + amplicons
ax_scatter_1 = fig.add_subplot(gs_a[0, 0])
plot_scatter_panel(ax_scatter_1, x_min=1, x_max=mid_pos, x_label=False)
ax_scatter_1.set_title("MAF ≥ 3 %", loc="center")

ax_amp_1 = fig.add_subplot(gs_a[1, 0], sharex=ax_scatter_1)
plot_amplicon_panel(ax_amp_1, x_min=1, x_max=mid_pos)
ax_amp_1.set_xlabel("Genome position", labelpad=0.8)

# Bottom half: scatter + amplicons
ax_scatter_2 = fig.add_subplot(gs_a[3, 0])
plot_scatter_panel(ax_scatter_2, x_min=mid_pos, x_max=max_pos, x_label=False)

ax_amp_2 = fig.add_subplot(gs_a[4, 0], sharex=ax_scatter_2)
plot_amplicon_panel(ax_amp_2, x_min=mid_pos, x_max=max_pos)
ax_amp_2.set_xlabel("Genome position", labelpad=0.8)

# Legend
handles_a, labels_a = ax_scatter_1.get_legend_handles_labels()
pair_list = sorted(zip(labels_a, handles_a), key=lambda x: x[0])
sorted_labels_a, sorted_handles_a = zip(*pair_list)
fig.legend(
    sorted_handles_a, sorted_labels_a,
    ncols=4,
    loc="upper center",
    markerscale=4,
    bbox_to_anchor=(0.5, 0.94),
    frameon=False
)

# add panel label
fig.text(0.03, 0.93, 'a', fontweight='bold', fontsize=8)


# Panel b: per-position histograms vs distance to nearest covering primer
ncols = 4
n_sites = len(sites)
nrows = int(np.ceil(n_sites / ncols))
gs_b = GridSpecFromSubplotSpec(
    nrows, ncols,
    subplot_spec=outer[1],
    wspace=0.27,
    hspace=0.41
)
axs_b = []

for i in range(nrows):
    for j in range(ncols):
        if not axs_b:
            ax = fig.add_subplot(gs_b[i, j])
            ref_ax = ax
        else:
            ax = fig.add_subplot(gs_b[i, j], sharey=ref_ax)
        axs_b.append(ax)

bins_prim = np.arange(0, max_dist_prim + bin_width, bin_width)
bar_width = bin_width *0.9

for idx, (ax, site) in enumerate(zip(axs_b, sites)):
    site_df = merged_isnv_dists[merged_isnv_dists["site"] == site].copy()
    row = idx // ncols
    col = idx % ncols

    for maf in maf_values:
        d = site_df[site_df["MAF"] == maf]
        if d.empty:
            continue

        d_unique = (
            d.drop_duplicates("Position")
             .dropna(subset=["min_primer_dist"])
        )
        if d_unique.empty:
            continue

        dist = d_unique["min_primer_dist"].to_numpy()
        counts, edges = np.histogram(dist, bins=bins_prim)

        bin_lefts = edges[:-1]

        ax.bar(
            bin_lefts,
            counts,
            width=bar_width,
            align ='edge',
            color=maf_to_color[maf],
            alpha=0.8,
            label=f"MAF ≥{maf}%" if idx == 0 else None
        )

    ax.set_title(site, pad=5)
    ax.grid(axis="x", linewidth=0.1, alpha=0.3)

    ticks = np.arange(0, max_dist_prim + 1, 10)
    ax.set_xticks(ticks)
    ax.set_xticklabels([t if t % 50 == 0 else "" for t in ticks])

    is_last_in_row = ((idx % ncols) == ncols - 1) or (idx == n_sites - 1)
    if row == nrows - 1 or is_last_in_row:
        ax.set_xlabel("Minimum distance\nto primer (bp)", labelpad=0.8)
    if col == 0:
        ax.set_ylabel("Number of\niSNV positions")

for ax in axs_b[len(sites):]:
    ax.set_visible(False)

handles_b, labels_b = axs_b[0].get_legend_handles_labels()
fig.legend(
    handles_b, labels_b,
    ncols=1,
    loc="upper center",
    markerscale=1,
    bbox_to_anchor=(0.81, 0.41),
    frameon=False
)
# add panel label
fig.text(0.03, 0.54, 'b', fontweight='bold', fontsize=8)

# panel c: scatter of proportion vs distance to amplicon end

gs_c = GridSpecFromSubplotSpec(
    nrows, ncols,
    subplot_spec=outer[2],
    wspace=0.27,
    hspace=0.41
)
axs_c = [fig.add_subplot(gs_c[i, j]) for i in range(nrows) for j in range(ncols)]

for idx, (ax, site) in enumerate(zip(axs_c, sites)):
    site_df = merged_isnv_dists[merged_isnv_dists["site"] == site].copy()
    row = idx // ncols
    col = idx % ncols

    for maf in maf_values:
        d = site_df[site_df["MAF"] == maf]
        if d.empty:
            continue

        d_unique = (
            d.drop_duplicates("Position")
             .dropna(subset=["min_dist_within_amplicon"])
        )
        if d_unique.empty:
            continue

        ax.scatter(
            d_unique["min_dist_within_amplicon"],
            d_unique["Proportion"],
            s=2,
            alpha=0.8,
            marker='o',
            edgecolors='none',
            color=maf_to_color[maf],
            label=f"MAF ≥{maf}%" if idx == 0 else None
        )

    ax.set_title(site, pad=5)
    ax.grid(axis="x", linewidth=0.1, alpha=0.3)

    ticks = np.arange(0, max_dist_amp + 1, 10)
    ax.set_xticks(ticks)
    ax.set_xticklabels([t if t % 50 == 0 else "" for t in ticks])

    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))

    is_last_in_row = ((idx % ncols) == ncols - 1) or (idx == n_sites - 1)
    if row == nrows - 1 or is_last_in_row:
        ax.set_xlabel("Minimum distance\n to amplicon end (bp)", labelpad=0.8)
    if col == 0:
        ax.set_ylabel("Proportion of samples\nwith iSNV")

for ax in axs_c[len(sites):]:
    ax.set_visible(False)

handles_c, labels_c = axs_c[0].get_legend_handles_labels()

fig.legend(
    handles_c, labels_c,
    ncols=1,
    loc="upper center",
    markerscale=2,
    bbox_to_anchor=(0.80, 0.17),
    frameon=False
)

# add panel label
fig.text(0.03, 0.30, 'c', fontweight='bold', fontsize=8)

# Save figure
plot_filename = 'figureS9_iSNV_primer_dists.pdf'
plot_path = os.path.join(figures_dir, "supplemental", plot_filename)
plt.savefig(plot_path, format='pdf', bbox_inches='tight')
plt.close()

print(f"Figure saved as '{plot_path}'")
