import os
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.dates import DateFormatter, MonthLocator
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pandas import Timedelta
from matplotlib.ticker import LogLocator, MaxNLocator, FuncFormatter


figures_dir = "../../figures/supplemental"

sample_metadata_path = "../../sample_protocol_list_with_dates_and_paper_lineage.csv"
samples_summary_path = "../../processed_data/samples_per_site_cov_summary.csv"
read_depths_path = "../../processed_data/10x/read_depth_summary.csv"

cov_thresh_label = "10x"
prop_depth_column = "samples_depth_1000_no_gaps"
mean_depth_column = "mean_read_depth_no_gaps"

selected_sites = [
    "NORT_ARTIC_3", "NORT_ARTIC_4", "NORT_ARTIC_4.1",
    "NORW_ARTIC_4.1", "OXON_VeSeq", "PHEC_ARTIC_3",
    "SANG_ARTIC_3", "SANG_ARTIC_4.1"
]

pretty_labels = {
    'NORT_ARTIC_3': 'NORT ARTIC 3',
    'NORT_ARTIC_4': 'NORT ARTIC 4',
    'NORT_ARTIC_4.1': 'NORT ARTIC 4.1',
    'NORW_ARTIC_4.1': 'NORW ARTIC ?',
    'OXON_VeSeq': 'OXON ve-SEQ',
    'PHEC_ARTIC_3': 'PHEC ARTIC 3',
    'SANG_ARTIC_3': 'SANG ARTIC 3',
    'SANG_ARTIC_4.1': 'SANG ARTIC 4.1'
}

def load_sample_totals(threshold=cov_thresh_label):
    file = samples_summary_path
    if not os.path.exists(file):
        print(f"Error: {file} not found"); sys.exit(1)
    df = pd.read_csv(file)
    df_filtered = df[df["Cov_Threshold"] == threshold].copy()
    df_filtered["site_short"] = df_filtered["Site"].str.replace(r"_ILLUMINA$", "", regex=True)
    return dict(zip(df_filtered["site_short"], df_filtered["Total_Samples"]))


# Load and prepare metadata for plotting panel a and b
df = pd.read_csv(
    sample_metadata_path,
    dtype={
        "sample_name": "string",
        "collection_date": "string",
        "sequencing_submission_date": "string",
        "protocol_primers": "string",
        "in_climb": "string",
        "paper_lineage": "string",
    }
)
if not os.path.exists(sample_metadata_path):
    print(f"Error: {sample_metadata_path} not found"); sys.exit(1)

# parse dates - collection_date is dmy, sequencing_submission_date is ymd
df["collection_date"] = pd.to_datetime(df["collection_date"], dayfirst=True, errors="coerce")
df["sequencing_submission_date"] = pd.to_datetime(df["sequencing_submission_date"], errors="coerce")

# Sequencing site from sample_name prefix (before first "-")
df["sequencing_site"] = df["sample_name"].str.extract(r"^([^-]+)").squeeze()

# Fill empty protocol_primers with "Unknown"
df["protocol_primers"] = df["protocol_primers"].fillna("").replace("", "Unknown")

# Filter away rows with entries in "in_climb" column
df = df[df["in_climb"].isna()].copy()

# Collapse SANG sites
df.loc[df["sequencing_site"].isin(["BRBR", "LSPA", "QEUH", "MILK"]), "sequencing_site"] = "SANG"

# Change "NORW ARTIC_Unknown" to "NORW ARTIC_?"
df.loc[(df["sequencing_site"] == "NORW") & (df["protocol_primers"] == "ARTIC_Unknown"), "protocol_primers"] = "ARTIC_?"

# Fix NORT ARTIC_4 date mis-match
mask_fix = (
    (df["sequencing_site"] == "NORT") &
    (df["protocol_primers"] == "ARTIC_4") &
    (df["sequencing_submission_date"] == pd.Timestamp("2021-01-06"))
)
df.loc[mask_fix, "sequencing_submission_date"] = pd.Timestamp("2022-01-06")

# Drop unwanted site–primer combos
unknowns = (
    (df["sequencing_site"].eq("SANG") & df["protocol_primers"].isin(["ARTIC_Unknown", "Unknown"])) |
    (df["sequencing_site"].eq("NORT") & df["protocol_primers"].eq("ARTIC_Unknown")) |
    (df["sequencing_site"].eq("PHEC") & df["protocol_primers"].eq("ARTIC_Unknown"))
)
df = df.loc[~unknowns].copy()

# Prepare data for collection and sequencing date span plot
g = df.groupby(["sequencing_site", "protocol_primers"], dropna=False)
span_df = g.agg(
    coll_start=("collection_date", "min"),
    coll_end=("collection_date", "max"),
    seq_start=("sequencing_submission_date", "min"),
    seq_end=("sequencing_submission_date", "max"),
).reset_index()

# Add track names and keys
span_df["track"] = span_df["sequencing_site"].astype(str) + " " + span_df["protocol_primers"].astype(str)
span_df["site_key"] = span_df["sequencing_site"] + "_" + span_df["protocol_primers"]
span_df["track_pretty"] = span_df["site_key"].map(pretty_labels).fillna(span_df["track"])

span_df = span_df.sort_values("coll_start", na_position="last").reset_index(drop=True)
span_df["track_num"] = np.arange(1, len(span_df) + 1, dtype=float)

def _to_long(df_):
    rows = []
    for _, r in df_.iterrows():
        # Two span types: collection and seq, each with start/end and a y offset between them
        rows.append({"track": r["track"], "track_num": r["track_num"], "span_type": "coll",
                     "start": r["coll_start"], "end": r["coll_end"], "y": r["track_num"] - 0.1})
        rows.append({"track": r["track"], "track_num": r["track_num"], "span_type": "seq",
                     "start": r["seq_start"], "end": r["seq_end"], "y": r["track_num"] + 0.1})
    return pd.DataFrame.from_records(rows)

span_long = _to_long(span_df)

# get total sample counts per site–primer combo
sample_summary = pd.read_csv(samples_summary_path)
sample_summary = sample_summary[sample_summary["Cov_Threshold"] == "10x"].copy()

no_platform = sample_summary["Site"].str.replace(r"_(ILLUMINA|OXFORD_NANOPORE)$", "", regex=True)
parts = no_platform.str.split("_", n=1, expand=True)
sample_summary["site_part"] = parts[0]
sample_summary["primer_part"] = parts[1]

# Build a key that matches site–primer combos to the track names
sample_summary["track"] = sample_summary["site_part"] + " " + sample_summary["primer_part"]
sample_summary = sample_summary[["track", "Total_Samples"]]

span_df["track_join"] = span_df["track"].where(
    ~span_df["track"].eq("NORW ARTIC_?"),
    "NORW ARTIC_4.1"
)

# Merge into span_df
span_df = span_df.merge(sample_summary.rename(columns={"track": "track_join"}), on="track_join", how="left").drop(columns=["track_join"])

# Primer release dates
primer_dates = pd.DataFrame({
    "name": ["ARTIC V3", "ARTIC V4", "ARTIC V4.1"],
    "date": pd.to_datetime(["2020-03-01", "2021-06-01", "2021-12-01"])
})
primer_colors = {
    "ARTIC V3": "royalblue",
    "ARTIC V4": "teal",
    "ARTIC V4.1": "forestgreen"
}


# Prepare data for lineage proportions plot (panel b)
df_plot = df.copy()
df_plot["paper_lineage"] = (
    df_plot["paper_lineage"]
      .replace({"Other":"Other/NA", "Unassingned":"Other/NA"})
      .fillna("Other/NA")
)

counts = (
    df_plot.groupby(["collection_date", "paper_lineage"])
           .size()
           .rename("n")
           .reset_index()
)
wide = counts.pivot_table(index="collection_date", columns="paper_lineage", values="n", fill_value=0).sort_index()

# Ensure "Other/NA" exists
if "Other/NA" not in wide.columns:
    wide["Other/NA"] = 0

row_sums = wide.sum(axis=1).replace(0, np.nan)
props = wide.div(row_sums, axis=0).fillna(0.0)

pref_order = ["B.1.177","B.1.1.7","B.1.351","P.1","P.2","B.1.525","B.1.617.2",
              "BA.1","BA.2","BA.3","BA.4","BA.5","BA.2.75","BQ.1","XBB","Other/NA"]
cols_present = [c for c in pref_order if c in props.columns]
props = props[cols_present]

# Colors
tab20 = mpl.colormaps.get_cmap("tab20")
non_other = [c for c in cols_present if c != "Other/NA"]
pal_non_other = {k: tab20(i) for i, k in enumerate(non_other)}
pal = {**pal_non_other}
if "Other/NA" in cols_present:
    pal["Other/NA"] ="lightgrey"

# Align date ranges for panels a & b
all_dates = pd.to_datetime(
    pd.Series(pd.concat([
        span_long["start"].dropna(),
        span_long["end"].dropna(),
        props.index.to_series()
    ])),
    errors="coerce"
).dropna()

xmin, xmax = all_dates.min(), all_dates.max()
xmin_padded = xmin - Timedelta(days=89)
xmax_padded = xmax + Timedelta(days=95)



# Load read depth data and prepare for plotting panel c and d
read_depths = pd.read_csv(read_depths_path)
if not os.path.exists(read_depths_path):
    print(f"Error: {read_depths_path} not found"); sys.exit(1)

read_depths_sel = read_depths[read_depths["Site"].isin(selected_sites)].copy()
site_groups = {site: df for site, df in read_depths_sel.groupby("Site")}
sample_totals = load_sample_totals(threshold=cov_thresh_label)


############################
# Plotting

plt.style.use("plot_style_settings.mplstyle")

fig = plt.figure(figsize=(7.5, 10.5), constrained_layout=True)
# Outer grid for all panels
outer = gridspec.GridSpec(
    nrows=4, ncols=1, height_ratios=[1.0, 1.0, 1.3, 1.3], hspace=0.4, figure=fig,
    left=0.12, right=0.88,
    bottom=0.04, top=0.95
)

### Panel a: collection and sequencing time spans
ax_a = fig.add_subplot(outer[0])

for _, r in span_long.dropna(subset=["start", "end"]).iterrows():
    color = "lightsteelblue" if r["span_type"] == "coll" else "steelblue"
    ax_a.plot([r["start"], r["end"]], [r["y"], r["y"]], color=color, lw=4)

# Add sample count labels to right of each track
pad_days = pd.Timedelta(days=10)
for _, r in span_df.iterrows():
    if pd.notna(r["Total_Samples"]) and pd.notna(r["seq_end"]):
        ax_a.text(
            r["seq_end"] + pad_days, r["track_num"],
            f"n = {int(r['Total_Samples'])}",
            va="center", ha="left", fontsize=6, color="black"
        )

# y labels as site–primer combos
ax_a.set_yticks(span_df["track_num"])
ax_a.set_yticklabels(span_df["track_pretty"])

# Primer release dates as vertical lines + labels
for _, pr in primer_dates.iterrows():
    color = primer_colors[pr["name"]]
    ax_a.axvline(pr["date"], ls="--", lw=0.4, color=color)
    ax_a.text(pr["date"], span_df["track_num"].max() + 0.6, pr["name"],
              rotation=30, ha="center", fontsize=6, color=color)

ax_a.set_xlim(xmin_padded, xmax_padded)
ax_a.xaxis.set_major_locator(MonthLocator(bymonthday=1, interval=3))
ax_a.xaxis.set_major_formatter(DateFormatter("%b %Y"))
for label in ax_a.get_xticklabels():
    label.set_rotation(30)
ax_a.set_xlabel("")
ax_a.set_ylabel("")
ax_a.grid(True, linestyle="--", linewidth=0.2, alpha=0.5)

# Legend
leg_a = ax_a.legend(
    handles=[
        Line2D([0],[0], color="lightsteelblue", lw=4, label="Collection"),
        Line2D([0],[0], color="steelblue", lw=4, label="Sequencing"),
    ],
    loc="upper left", bbox_to_anchor=(1.01, 0.95), borderaxespad=0.
)


### Panel b: lineage proportions
ax_b = fig.add_subplot(outer[1])

legend_order = cols_present
draw_order = legend_order[::-1]
x = props.index.values
ys = [props[c].values for c in draw_order]
cols_draw = [pal[c] for c in draw_order]
ax_b.stackplot(x, ys, colors=cols_draw, baseline="zero")

ax_b.set_xlim(xmin_padded, xmax_padded)
ax_b.set_ylim(0, 1)
ax_b.xaxis.set_major_locator(MonthLocator(bymonthday=1, interval=3))
ax_b.xaxis.set_major_formatter(DateFormatter("%b %Y"))
for label in ax_b.get_xticklabels():
    label.set_rotation(30)
ax_b.set_ylabel("Proportion of samples")
ax_b.grid(True, axis="y", linestyle="--", linewidth=0.2, alpha=0.5)

legend_handles = [Patch(facecolor=pal[name], label=name) for name in legend_order]
leg_b = ax_b.legend(handles=legend_handles, ncol=1, loc="upper left", title="Lineage", bbox_to_anchor=(1.01, 1.2))


### Panel c: mean read depth
n_sites = len(selected_sites)
cols = 4
rows = (n_sites + cols - 1) // cols

grid_c = gridspec.GridSpecFromSubplotSpec(rows, cols, subplot_spec=outer[2], wspace=0.1, hspace=0.3)

for idx, site in enumerate(selected_sites):
    r, c = divmod(idx, cols)
    ax = fig.add_subplot(grid_c[r, c])
    df_site = site_groups[site]

    # Site label
    site_label = pretty_labels.get(site, site)
    ax.text(0.5, 1.08, site_label, transform=ax.transAxes, ha="center", va="bottom", fontsize=6)

    # Mean read depth curve
    ax.plot(df_site["Position"], df_site[mean_depth_column], color="steelblue")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_ylim(0.2, 1e5)
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=5))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{int(y):,}" if y >= 1 else "0"))
    ax.grid(True, which="major", linewidth=0.1, linestyle="--")

    # Spines and labels
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if c == 0:
        ax.set_ylabel("Mean read depth")
    else:
        ax.set_ylabel("")
        ax.set_yticklabels([])
    if r == rows - 1:
        ax.set_xlabel("Genome position")
    else:
        ax.set_xlabel("")
        ax.set_xticklabels([])


### Panel d: proportion ≥ threshold
grid_d = gridspec.GridSpecFromSubplotSpec(rows, cols, subplot_spec=outer[3], wspace=0.1, hspace=0.3)

for idx, site in enumerate(selected_sites):
    r, c = divmod(idx, cols)
    ax = fig.add_subplot(grid_d[r, c])
    df_site = site_groups[site]

    total = sample_totals.get(site, np.nan)
    if np.isnan(total):
        raise ValueError(f"No total-samples entry for site {site!r} at threshold {cov_thresh_label}")

    prop = df_site[prop_depth_column] / total

    # Site label
    site_label = pretty_labels.get(site, site)
    ax.text(0.5, 1.08, site_label, transform=ax.transAxes, ha="center", va="bottom", fontsize=6)

    # Proportion polygon
    ax.fill_between(df_site["Position"], prop, color="lightsteelblue", alpha=0.8, linewidth=0.4)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.grid(True, which="major", linewidth=0.1, linestyle="--")

    # Spines and labels
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if c == 0:
        ax.set_ylabel(f"Prop. samples\n≥1000×")
    else:
        ax.set_ylabel("")
        ax.set_yticklabels([])
    if r == rows - 1:
        ax.set_xlabel("Genome position")
    else:
        ax.set_xlabel("")
        ax.set_xticklabels([])

# Panel labels
fig.text(0.02, 0.98, "a", fontsize=8, fontweight="bold")
fig.text(0.02, 0.75, "b", fontsize=8, fontweight="bold")
fig.text(0.02, 0.52, "c", fontsize=8, fontweight="bold")
fig.text(0.02, 0.26, "d", fontsize=8, fontweight="bold")


# Save figure
outname = "figureS1_sample_sequence_read_depth_overview.png"
outpath = os.path.join(figures_dir, outname)
plt.savefig(outpath, format="png")
plt.close()
print(f"[Success] Saved {outpath}")
