##############################
# This script analyses the frequency of iSNVs at masked positions over time in the ONS-CIS study.

# This script processes variant data from multiple sequencing centres / protocols (sites) at masked positions with a read depth threshold > 1000x.
# It returns the number of samples with variants above certain MAF thresholds at each position
# and the total number of samples with sufficient depth at each position, each aggregated by week.

# The outputs are written to CSV files for further analysis to make figure S7, /scripts/plots/plot_figureS7_temporal_distribution.R
    # final_temporal_summary.csv
    # weekly_sample_count_per_site.csv
##############################

import pandas as pd
import numpy as np
import pyarrow.dataset as ds
import os
import re
from pathlib import Path

# --- Define Paths ---
base_dir = "/processed_data/10x/"
out_dir = "/processed_data/S7_temporal_analysis/"
os.makedirs(out_dir, exist_ok=True)

ons_summary_file = "dummy_ons_summary.csv"
mask_file_path = os.path.join(base_dir, "final_masks_all_centres.csv")
threshold_file_path = os.path.join(base_dir, "final_maf_thresholds.csv")

# --- 1. Load Sequence Metadata ---
print("Loading metadata...")
ons_summary = pd.read_csv(ons_summary_file, usecols=["sample_name", "collection_date"])
print(f"ons_summary head:\n{ons_summary.head()}")

print(f"ons_summary shape: {ons_summary.shape}")
maf_data = pd.read_csv(threshold_file_path)
mask_data = pd.read_csv(mask_file_path)

# --- 2. Parse Mask Positions ---
def parse_mask_string(mask_str):
    if pd.isna(mask_str) or mask_str == "[]":
        return []
    clean = re.sub(r"[\[\]'\"\s]", "", str(mask_str))
    if clean == "":
        return []
    return [int(float(x)) for x in clean.split(",") if x]

all_masked_positions_raw = mask_data[
    mask_data["mask"] != "positions_in_over_20pct_masks"
]["positions"]

masked_positions = set()
for s in all_masked_positions_raw:
    masked_positions.update(parse_mask_string(s))

print(f"Masked positions count: {len(masked_positions)}")

thresholds = maf_data["MAF_analysis"].unique()
unique_sites = maf_data["Site"].unique()

# --- 3. Site Folder Map ---
site_folder_map = {
    "SANG_ARTIC_4.1": ["BRBR_ARTIC_4.1_ILLUMINA", "LSPA_ARTIC_4.1_ILLUMINA", "QEUH_ARTIC_4.1_ILLUMINA"],
    "SANG_ARTIC_3":   ["MILK_ARTIC_3_ILLUMINA", "QEUH_ARTIC_3_ILLUMINA"],
    "NORW_ARTIC_4.1": ["NORW_ARTIC_Unknown_ILLUMINA"],
    "NORT_ARTIC_3":   ["NORT_ARTIC_3_ILLUMINA"],
    "NORT_ARTIC_4":   ["NORT_ARTIC_4_ILLUMINA"],
    "NORT_ARTIC_4.1": ["NORT_ARTIC_4.1_ILLUMINA"],
    "OXON_VeSeq":     ["OXON_VeSeq_ILLUMINA"],
    "PHEC_ARTIC_3":   ["PHEC_ARTIC_3_ILLUMINA"],
}

# --- 4. Parse collection dates once ---
ons_summary["date_parsed"] = pd.to_datetime(
    ons_summary["collection_date"], format="%d/%m/%Y", errors="coerce"
)
ons_summary["collection_week"] = ons_summary["date_parsed"].dt.to_period("W").dt.start_time
ons_summary = ons_summary.dropna(subset=["collection_week"])

# --- 5. Main Loop ---
global_weekly_depth = {}
global_sample_counts = {}

for site in unique_sites:
    print(f"\nProcessing Site: {site}")

    if site not in site_folder_map:
        print(f"  No folder mapping for: {site}")
        continue

    target_folders = site_folder_map[site]
    variant_files = [
        os.path.join(base_dir, folder, "allele_freqs_output.parquet")
        for folder in target_folders
    ]
    variant_files = [f for f in variant_files if os.path.exists(f)]

    if not variant_files:
        print(f"  No parquet files found for: {site}")
        continue

    try:
        # Open dataset
        dataset = ds.dataset(variant_files, format="parquet")
        print("  [1] Dataset opened")

        # --- Sample counts ---
        df_samples = dataset.to_table(
            columns=["sample"]
        ).to_pandas()
        df_samples = df_samples.rename(columns={"sample": "sample_name"})
        df_samples["sample_name"] = df_samples["sample_name"].astype(str)
        all_samples = df_samples["sample_name"].unique()
        print(f"  [2] Distinct samples: {len(all_samples)}")

        site_sample_counts = (
            pd.DataFrame({"sample_name": all_samples})
            .merge(ons_summary, on="sample_name", how="inner")
            .groupby("collection_week")
            .agg(n_samples=("sample_name", "nunique"))
            .reset_index()
        )
        site_sample_counts["Site"] = site
        global_sample_counts[site] = site_sample_counts
        print("  [3] Sample counts summarised")

        # --- Load full site data once (depth filtered + position filtered) ---
        print("  [4] Loading full site data...")

        # Filter to masked positions and depth > 1000 directly in the dataset query to minimize data read
        # Optimized Step 4
        df_full = dataset.to_table(
            columns=["sample", "pos", "read_depth_no_gaps", "min_allele_freq_no_gaps"],
            filter=(ds.field("read_depth_no_gaps") > 1000) & (ds.field("pos").isin(list(masked_positions)))
        ).to_pandas()

        df_full = df_full.rename(columns={
            "sample":                   "sample_name",
            "pos":                      "Position",
            "read_depth_no_gaps":       "read_depth",
            "min_allele_freq_no_gaps":  "maf"
        })
        df_full["sample_name"] = df_full["sample_name"].astype(str)

        print(f"  [5] Rows after depth + position filter: {len(df_full)}")
        
        print(f"  [DIAG] df_full sample example before merge: {repr(df_full['sample_name'].iloc[0])}")

        if df_full.empty:
            print("  No data after filtering, skipping site.")
            continue

        print(f" [DIAG] ons_summary : {ons_summary.head()}")
        print(f"  [DIAG] ons_summary shape: {ons_summary.shape}")

        # Join with ONS metadata
        df_full = df_full.merge(ons_summary, on="sample_name", how="inner")

        print(f"  [6] All samples for batching: {len(all_samples)}")

        # --- Batch over samples ---
        batch_size = 500
        sample_batches = [
            all_samples[i:i + batch_size]
            for i in range(0, len(all_samples), batch_size)
        ]

        site_results = []

        # Before the batch loop, ensure types match
        df_full["sample_name"] = df_full["sample_name"].astype(str).str.strip()
        all_samples = np.array([str(s).strip() for s in all_samples])

        print(f" df_full : {df_full.head()}")
        print(f"  df_full sample_name dtype: {df_full['sample_name'].dtype}")
        print(f"  all_samples dtype: {all_samples.dtype}")
        print(f"  all_samples: {all_samples[:5]}")
        print(f"  Overlap count: {df_full['sample_name'].isin(all_samples).sum()}")

        for batch_i, batch_samples in enumerate(sample_batches):
            print(f"  Batch {batch_i + 1} of {len(sample_batches)}")

            batch_samples_set = set(batch_samples)
            batch_data = df_full[df_full["sample_name"].isin(batch_samples_set)].copy()
            print(f"    Rows in batch: {len(batch_data)}")

            if batch_data.empty:
                continue

            # Denominators
            batch_denominators = (
                batch_data
                .groupby(["collection_week", "Position"])
                .agg(sample_count_depth_1000x=("sample_name", "nunique"))
                .reset_index()
            )

            # Threshold hits
            threshold_hits_list = []
            for thr in thresholds:
                hits = (
                    batch_data[batch_data["maf"] > (thr / 100)]
                    .groupby(["collection_week", "Position"])
                    .agg(count=("sample_name", "nunique"))
                    .reset_index()
                )
                hits[f"hits_at_{thr}_pct"] = hits["count"]
                hits = hits.drop(columns=["count"])
                threshold_hits_list.append(hits)

            # Merge all threshold hit columns
            batch_threshold_hits = threshold_hits_list[0]
            for hits_df in threshold_hits_list[1:]:
                batch_threshold_hits = batch_threshold_hits.merge(
                    hits_df, on=["collection_week", "Position"], how="outer"
                )
            hits_cols = [c for c in batch_threshold_hits.columns
                         if c.startswith("hits_at_")]
            batch_threshold_hits[hits_cols] = batch_threshold_hits[hits_cols].fillna(0)

            batch_result = batch_denominators.merge(
                batch_threshold_hits, on=["collection_week", "Position"], how="left"
            )
            hits_cols = [c for c in batch_result.columns if c.startswith("hits_at_")]
            batch_result[hits_cols] = batch_result[hits_cols].fillna(0)

            site_results.append(batch_result)

        if not site_results:
            print("  No data collected for site.")
            continue

        # Aggregate batches
        site_summary = pd.concat(site_results)
        hits_cols = [c for c in site_summary.columns if c.startswith("hits_at_")]
        site_summary = (
            site_summary
            .groupby(["collection_week", "Position"])
            .agg(
                sample_count_depth_1000x=("sample_count_depth_1000x", "sum"),
                **{col: (col, "sum") for col in hits_cols}
            )
            .reset_index()
        )
        site_summary["Site"] = site
        global_weekly_depth[site] = site_summary
        print(f"  Done: {len(site_summary)} rows")

    except Exception as e:
        print(f"  Error in site {site}: {e}")
        import traceback
        traceback.print_exc()

# --- 6. Final Aggregation ---
if not global_weekly_depth:
    print("No data collected across any site. Exiting.")
    exit(1)

print("\nAggregating final summary...")
all_site_data = pd.concat(global_weekly_depth.values())
hits_cols = [c for c in all_site_data.columns if c.startswith("hits_at_")]

final_temporal_summary = (
    all_site_data
    .groupby(["collection_week", "Position"])
    .agg(
        total_samples_1000x=("sample_count_depth_1000x", "sum"),
        labs_reporting=("Site", "nunique"),
        **{col: (col, "sum") for col in hits_cols}
    )
    .reset_index()
)

sample_counts_long = pd.concat(global_sample_counts.values())

# --- 7. Write Outputs ---
final_temporal_summary.to_csv(
    os.path.join(out_dir, "temporal_summary.csv"), index=False
)
sample_counts_long.to_csv(
    os.path.join(out_dir, "weekly_sample_count_per_site.csv"), index=False
)

print("\nDone! Output written to:")
print(f"  {os.path.join(out_dir, 'temporal_summary.csv')}")
print(f"  {os.path.join(out_dir, 'weekly_sample_count_per_site.csv')}")
