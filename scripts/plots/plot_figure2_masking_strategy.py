import os
import ast
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import matplotlib.ticker as mtick
import dask.dataframe as dd
from math import ceil

base_processed_dir = "../../processed_data/"
figures_dir = "../../figures/"

site_to_folder_map = {
    'NORT_ARTIC_3': 'NORT_ARTIC_3_ILLUMINA',
    'NORT_ARTIC_4': 'NORT_ARTIC_4_ILLUMINA',
    'NORT_ARTIC_4.1': 'NORT_ARTIC_4.1_ILLUMINA',
    'NORW_ARTIC_4.1': 'NORW_ARTIC_4.1_ILLUMINA',
    'OXON_VeSeq': 'OXON_VeSeq_ILLUMINA',
    'PHEC_ARTIC_3': 'PHEC_ARTIC_3_ILLUMINA',
    'SANG_ARTIC_3': 'SANG_ARTIC_3_ILLUMINA',
    'SANG_ARTIC_4.1': 'SANG_ARTIC_4.1_ILLUMINA'
}
sites = list(site_to_folder_map.keys())

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
shared_percentages = [f"{x:.1f}%shared" for x in np.arange(0.5, 20.5, 0.5)]
all_mafs = list(range(2, 21))


isnvs_path = os.path.join(base_processed_dir, '10x/total_isnvs_per_sample_and_maf_all_masks_depth_1000x/*.parquet')
ddf = dd.read_parquet(
    isnvs_path,
    engine='pyarrow',
    columns=['ID', 'Site','Mask_Condition','MAF_Threshold_Percent','iSNV_Count']
).categorize(columns=['Site','Mask_Condition'])

mask_path = os.path.join(base_processed_dir,'10x/final_masks_all_centres.csv')
mask_df = pd.read_csv(mask_path)
mask_df = mask_df[mask_df['mask'].str.startswith('1.5%shared_')]
best_masks = mask_df.set_index('Centre_protocol')['mask'].to_dict()

best_unmasked_maf_per_site = {
    site: int(mask_str.split('_')[1].replace('%MAF', ''))
    for site, mask_str in best_masks.items()
}

# Panel a and c: Average iSNVs per sample and MAF threshold
mean_df = (
    ddf.groupby(['Site','Mask_Condition','MAF_Threshold_Percent'], observed=True)
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

# Reference (unmasked OXON at 3% MAF)
thresholds = sorted(pivoted.index)
ref_site = 'OXON_VeSeq'
ref_thresh = 3
ref_y_value = pivoted[(ref_site, 'no_mask')].reindex(thresholds).loc[ref_thresh]


# Panel b data: masked positions per MAF threshold
all_masks = []
for site, folder in site_to_folder_map.items():
    all_masks_path = os.path.join(base_processed_dir, '10x', folder, 'mask_files_depth_1000x', 'all_masks.csv')
    df_site_masks = pd.read_csv(all_masks_path)
    df_site_masks['Centre_protocol'] = site
    all_masks.append(df_site_masks)
mask_df_full = pd.concat(all_masks, ignore_index=True)

shared_df = mask_df_full[mask_df_full['mask_file'].str.startswith('1.5%shared_')].copy()
shared_df['MAF'] = shared_df['mask_file'].str.extract(r'_(\d+)%MAF').astype(int)
shared_df['Masked_Positions'] = shared_df['positions'].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else []
)
shared_df['Masked_Count'] = shared_df['Masked_Positions'].apply(len)

shared_df['Selected'] = shared_df.apply(
    lambda row: row['MAF'] == best_unmasked_maf_per_site.get(row['Centre_protocol'], -1),
    axis=1
)


# Panel c: compute best masked MAF per site
best_masked_maf_per_site = {}
for site, mask_name in best_masks.items():
    y_masked = pivoted[(site, mask_name)].reindex(thresholds).values
    i = int(np.nanargmin(np.abs(y_masked - ref_y_value)))
    best_masked_maf_per_site[site] = thresholds[i]

needed_rows = []
for site in sites:
    fixed_MAF = 3
    best_mask = best_masks[site]
    best_maf = best_unmasked_maf_per_site[site]
    ref_maf = best_masked_maf_per_site[site]
    needed_rows.extend([
        (site, 'no_mask', fixed_MAF),
        (site, best_mask, best_maf),
        (site, best_mask, ref_maf)
    ])

needed_df = pd.DataFrame(list(set(needed_rows)), columns=['Site', 'Mask_Condition', 'MAF_Threshold_Percent'])
needed_df['Site'] = pd.Categorical(needed_df['Site'], categories=ddf['Site'].cat.categories, ordered=False)
needed_df['Mask_Condition'] = pd.Categorical(needed_df['Mask_Condition'], categories=ddf['Mask_Condition'].cat.categories, ordered=False)

ddf_filtered = ddf[['ID','Site','Mask_Condition','MAF_Threshold_Percent','iSNV_Count']].merge(
    needed_df, on=['Site','Mask_Condition','MAF_Threshold_Percent']
)
df_filtered = ddf_filtered.compute()

sample_totals_path = os.path.join(base_processed_dir, "samples_per_site_cov_summary.csv")
sample_totals = (
    pd.read_csv(sample_totals_path)
    .query("Cov_Threshold == '10x'")
    .set_index("Site")['Total_Samples']
    .to_dict()
)

###################################
### Plotting ###

plt.style.use("plot_style_settings.mplstyle")

ncols = 4
nrows_grid = (len(sites) + ncols - 1) // ncols

fig = plt.figure(figsize=(7.5, 9.0))
outer = GridSpec(3, 1, height_ratios=[1.0, 0.35, 1.0], hspace=0.39)


### Panel a: average iSNVs per sample
gs_a = GridSpecFromSubplotSpec(
    nrows_grid, ncols,
    subplot_spec=outer[0],
    hspace=0.3, wspace=0.08
)
axs_a = [fig.add_subplot(gs_a[i, j]) for i in range(nrows_grid) for j in range(ncols)]

best_masked_maf_per_site_from_a = {}

for idx, (ax, site) in enumerate(zip(axs_a, sites)):
    row, col = divmod(idx, ncols)

    # Unmasked curve
    y_no = pivoted[(site, "no_mask")].reindex(thresholds).values
    ax.plot(
        thresholds, y_no,
        "-o",
        color="darkgrey",
        linewidth=0.6,
        markersize=1.4,
        label="no mask",
    )

    # Unmasked reference lines
    diffs_un = np.abs(y_no - ref_y_value)
    closest_un_idx = int(np.nanargmin(diffs_un))
    ax.axvline(x=thresholds[closest_un_idx], linestyle=":", color="steelblue", linewidth=0.9)
    ax.axhline(y=ref_y_value, linestyle=":", color="darkgrey", linewidth=0.9)

    # Best mask curve + analysis MAF vertical line
    bm = best_masks.get(site)
    if bm and (site, bm) in pivoted.columns:
        ybm = pivoted[(site, bm)].reindex(thresholds).values

        maf_val = bm.split("_")[1].replace("%MAF", "")

        ax.plot(
            thresholds, ybm,
            "-o",
            color="black",
            linewidth=0.6,
            markersize=1.4,
        )

        diffs_bm = np.abs(ybm - ref_y_value)
        closest_bm_idx = int(np.nanargmin(diffs_bm))
        best_maf_masked = thresholds[closest_bm_idx]
        best_masked_maf_per_site_from_a[site] = best_maf_masked
        ax.axvline(x=best_maf_masked, linestyle=":", color="darkorange", linewidth=0.9)

    ax.set_title(pretty_labels.get(site, site))
    ax.set_xticks(all_mafs)
    ax.tick_params(axis="both", which="both", bottom=True, left=True)
    ax.set_xticklabels([m if m % 2 == 0 else "" for m in all_mafs])
    ax.set_ylim(-0.5, 40)
    ax.grid(True)

    if col == 0:
        ax.set_ylabel("Mean iSNVs per sample")
    else:
        ax.yaxis.set_tick_params(labelleft=False)

    if row == nrows_grid - 1:
        ax.set_xlabel("MAF threshold (%)")

handles_a = [
    plt.Line2D([0], [0], color="darkgrey", lw=0.6, marker="o", markersize=2, label="no mask"),
    plt.Line2D([0], [0], color="black", lw=0.6, marker="o", markersize=2, label="masked"),
    plt.Line2D([0], [0], color="steelblue", lw=0.9, linestyle=":",
           label="masking MAF (no mask ≈10 iSNVs)"),
    plt.Line2D([0], [0], color="darkorange", lw=0.9, linestyle=":",
           label="analysis MAF (masked ≈10 iSNVs)"),
]

fig.legend(
    handles=handles_a,
    ncols=4,
    loc="upper right",
    bbox_to_anchor=(0.885, 0.925),
    bbox_transform=fig.transFigure,
    frameon=False
)

# Hide unused subplots
for ax in axs_a[len(sites):]:
    ax.axis("off")


### Panel b: masked positions per MAF threshold
gs_b = GridSpecFromSubplotSpec(2, 8, subplot_spec=outer[1], height_ratios=[0.35, 1], hspace=0.1, wspace=0.42)
ncols_b = 8
for idx, site in enumerate(sites):
    row, col = divmod(idx, ncols_b)
    top_idx, bottom_idx = (0, 1) if row == 0 else (3, 4)

    ax_top = fig.add_subplot(gs_b[top_idx, col])
    ax_bottom = fig.add_subplot(gs_b[bottom_idx, col], sharex=ax_top)

    df_site = (
        shared_df[shared_df['Centre_protocol'] == site]
        .set_index('MAF')
        .reindex(all_mafs)
        .reset_index()
        .assign(
            Centre_protocol=site,
            Masked_Count=lambda df: df['Masked_Count'].fillna(0),
            Selected=lambda df: df['Selected'].fillna(False)
        )
    )

    for ax in [ax_top, ax_bottom]:
        sns.barplot(
            data=df_site,
            x='MAF',
            y='Masked_Count',
            hue='Selected',
            palette={True: 'steelblue', False: 'darkgrey'},
            dodge=False,
            width=0.7,
            edgecolor = 'none',
            ax=ax,
        )
        ax.get_legend().remove()


    ax_bottom.set_ylim(0, 150)
    top_values = df_site['Masked_Count'][df_site['Masked_Count'] > 150]
    top_min = 151
    top_max = ceil((top_values.max() * 1.05) / 100) * 100 if not top_values.empty else 200
    ax_top.set_ylim(top_min, top_max)

    # Add count labels only to selected bars
    selected_rows = df_site[df_site['Selected']].copy()
    x_offset = 1.5  # move labels slightly to the right

    for _, selected_row in selected_rows.iterrows():
        selected_maf = selected_row['MAF']
        selected_count = selected_row['Masked_Count']

        x_pos = df_site.index[df_site['MAF'] == selected_maf][0]

        if selected_count <= 150:
            label_ax = ax_bottom
        else:
            label_ax = ax_top

        label_ax.text(
            x_pos + x_offset,
            selected_count,
            f"{int(selected_count)}",
            ha='center',
            va='bottom',
            fontsize=6,
            clip_on=True
        )

     # break styling
    ax_top.spines["bottom"].set_linestyle("--")
    ax_top.spines["bottom"].set_color("grey")
    ax_top.spines["bottom"].set_linewidth(0.3)
    ax_bottom.spines["top"].set_linestyle("--")
    ax_bottom.spines["top"].set_color("grey")
    ax_bottom.spines["top"].set_linewidth(0.3)
    ax_top.tick_params(bottom=False, labelbottom=False)

    # tidy bottom xticks to show every other label
    xticks_bottom = ax_bottom.get_xticks()
    xtick_labels_bottom = [label.get_text() for label in ax_bottom.get_xticklabels()]
    ax_bottom.set_xticks(xticks_bottom)
    ax_bottom.set_xticklabels([label if i % 4 == 0 else '' for i, label in enumerate(xtick_labels_bottom)], ha='center')
    ax_bottom.tick_params(axis='y', pad=1, labelsize=6)
    ax_top.tick_params(axis='y', pad=1, labelsize=6)
    ax_bottom.tick_params(axis='x', pad=2, labelsize=6.5)
    for i, tick in enumerate(ax_bottom.xaxis.get_major_ticks()):
        if i % 4 == 0:
            tick.tick1line.set_markersize(2.5)
        else:
            tick.tick1line.set_markersize(1.5)

    ax_top.set_title(pretty_labels.get(site, site), pad=5, fontsize=6.5)
    ax_bottom.set_ylabel('')
    ax_top.set_ylabel('')
    ax_top.tick_params(axis='both', which='both', bottom=False, left=True)
    ax_bottom.tick_params(axis='both', which='both', bottom=True, left=True)
    ax_top.grid(True, axis='y', which='both')
    ax_bottom.grid(True, axis='y', which='both')

    if col == 0:
        fig.text(0.08, (ax_bottom.get_position().y0 + ax_top.get_position().y1) / 2,
                 'Masked positions (n)',
                 va='center', ha='center', rotation='vertical')
    if col == ncols_b // 2:
        ax_bottom.set_xlabel('MAF threshold (%)              ', loc='right' )
    else:
        ax_bottom.set_xlabel('')

# Remove unused subplots
for idx in range(len(sites), nrows_grid * ncols):
    for i in [0, 1, 3, 4]:
        fig.add_subplot(gs_b[i, idx % ncols]).axis('off')


### Panel c: histograms of iSNVs per sample
gs_c = GridSpecFromSubplotSpec(nrows_grid, ncols, subplot_spec=outer[2], hspace=0.3, wspace=0.08)
axs_c = [fig.add_subplot(gs_c[i, j]) for i in range(nrows_grid) for j in range(ncols)]

color_unmasked = 'darkgrey'
color_masked_ref = 'steelblue'
color_masked_best = 'darkorange'

for idx, site in enumerate(sites):
    ax = axs_c[idx]
    ax.set_title(pretty_labels.get(site, site))

    subsets = [
        {'label': 'Unmasked (3% MAF)', 'color': color_unmasked, 'mask': 'no_mask', 'maf': 3},
        {'label': 'Masked (unmasked MAF threshold)', 'color': color_masked_ref, 'mask': best_masks[site], 'maf': best_unmasked_maf_per_site[site]},
        {'label': 'Masked (masked MAF threshold)', 'color': color_masked_best, 'mask': best_masks[site], 'maf': best_masked_maf_per_site[site]},
    ]

    long_site = site_to_folder_map.get(site)
    total = int(sample_totals.get(long_site, 0))

    for s in subsets:
        subset_df = df_filtered[
            (df_filtered['Site'] == site) &
            (df_filtered['Mask_Condition'] == s['mask']) &
            (df_filtered['MAF_Threshold_Percent'] == s['maf'])
        ]
        counts = subset_df['iSNV_Count'].to_numpy(dtype=int)

        # Pad with zeros for samples with no calls (so proportions are out of total samples)
        if total > 0 and len(counts) < total:
            counts = np.concatenate([counts, np.zeros(total - len(counts), dtype=int)])
        elif total == 0:
            # If sample total is unknown, fall back to observed counts
            total = len(counts) if len(counts) > 0 else 1

        weights = np.ones_like(counts) / total

        ax.hist(
            counts,
            bins=np.arange(0, 41),
            weights=weights,
            color=s['color'],
            label=s['label'],
            alpha=0.8,
            edgecolor='black',
            linewidth=0.2
        )

        # CDF curve
        if counts.size > 0:
            full_bins = np.arange(0, counts.max() + 2)
            hist, bin_edges = np.histogram(counts, bins=full_bins)
            cdf = np.cumsum(hist) / total
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            plot_mask = bin_centers <= 40
            ax.plot(
                bin_centers[plot_mask],
                cdf[plot_mask],
                color=s['color'],
                linestyle='--',
                linewidth=0.6
            )

    ax.set_xlim(0, 40)
    ax.set_xticks(np.arange(0, 41, 10))
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis='both', which='both', bottom=True, left=True)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
    ax.grid(True)
    if idx % ncols == 0:
        ax.set_ylabel("Proportion of samples")
    else:
        ax.yaxis.set_tick_params(labelleft=False)
    if idx >= ncols * (nrows_grid - 1):
        ax.set_xlabel("iSNVs per sample")

# Hide unused subplots
for ax in axs_c[len(sites):]:
    ax.axis('off')

# Legend for panel c
handles = [
    plt.Line2D([0], [0], color=color_unmasked, lw=4, solid_capstyle='butt', label='no mask (MAF ≥ 3%)'),
    plt.Line2D([0], [0], color=color_masked_ref, lw=4, solid_capstyle='butt', label='masked (masking MAF)'),
    plt.Line2D([0], [0], color=color_masked_best, lw=4, solid_capstyle='butt', label='masked (analysis MAF)'),
]


panel_c_top = axs_c[0].get_position().y1

fig.legend(handles=handles,
           loc="upper center",
           bbox_to_anchor=(0.5, panel_c_top + 0.04),
           ncol=3,
           frameon=False)

### 
# Panel labels
fig.text(0.07, 0.92, 'a', fontweight='bold', fontsize=8)
fig.text(0.07, 0.57, 'b', fontweight='bold', fontsize=8)
fig.text(0.07, 0.40, 'c', fontweight='bold', fontsize=8)

# Save figure
plot_filename = 'figure2_masking_strategy.pdf'
plot_path = os.path.join(figures_dir, plot_filename)
plt.savefig(plot_path, format='pdf', dpi=600, bbox_inches='tight')
plt.close()

print(f"Figure saved as '{plot_path}'")

# Write final MAF thresholds to CSV
summary_rows = []
for site in sites:
    summary_rows.append({
        "Site": site,
        "mask_name": best_masks[site],
        "MAF_mask": int(best_unmasked_maf_per_site[site]),
        "MAF_analysis": int(best_masked_maf_per_site[site])
    })
maf_summary_df = pd.DataFrame(summary_rows, columns=["Site", "mask_name", "MAF_mask", "MAF_analysis"])
summary_out_dir = os.path.join(base_processed_dir, "10x", "final_maf_thresholds.csv")
maf_summary_df.to_csv(summary_out_dir, index=False)
print(f"Wrote final MAF thresholds to: {summary_out_dir}")
