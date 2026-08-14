##############################
# Get daily sample counts per sequencing site
##############################

import pandas as pd
import pyarrow.dataset as ds
import os

base_dir = "/processed_data/10x/"
out_dir = "/processed_data/S7_temporal_analysis/"
os.makedirs(out_dir, exist_ok=True)

ons_summary_file = "dummy_ons_summary.csv"

# Load Sequence Metadata
ons_summary = pd.read_csv(ons_summary_file, usecols=["sample_name", "collection_date"])

ons_summary["collection_date"] = pd.to_datetime(
    ons_summary["collection_date"], format="%d/%m/%Y", errors="coerce"
).dt.strftime('%Y-%m-%d')
ons_summary = ons_summary.dropna(subset=["collection_date"])

# Site Folder Map
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

# Main Loop
global_sample_counts = []

for site, target_folders in site_folder_map.items():
    variant_files = [
        os.path.join(base_dir, folder, "allele_freqs_output.parquet")
        for folder in target_folders
    ]
    variant_files = [f for f in variant_files if os.path.exists(f)]

    if not variant_files:
        print(f"  No parquet files found for: {site}")
        continue

    try:
        dataset = ds.dataset(variant_files, format="parquet")
        
        # Extract unique samples for this site
        df_samples = dataset.to_table(columns=["sample"]).to_pandas()
        df_samples = df_samples.rename(columns={"sample": "sample_name"})
        df_samples["sample_name"] = df_samples["sample_name"].astype(str)
        all_samples = df_samples["sample_name"].unique()
      
        # Merge with metadata and group by DAY (collection_date)
        site_sample_counts = (
            pd.DataFrame({"sample_name": all_samples})
            .merge(ons_summary, on="sample_name", how="inner")
            .groupby("collection_date")
            .agg(n_samples=("sample_name", "nunique"))
            .reset_index()
        )
        site_sample_counts["Site"] = site
        
        global_sample_counts.append(site_sample_counts)

    except Exception as e:
        print(f"  Error in site {site}: {e}")

# Final Aggregation & Output ---
if not global_sample_counts:
    print("No data collected across any site. Exiting.")
    exit(1)

final_counts_df = pd.concat(global_sample_counts, ignore_index=True).sort_values(by=["Site", "collection_date"])[["collection_date", "n_samples", "Site"]]

# Write the CSV file
output_file = os.path.join(out_dir, "daily_sample_count_per_site.csv")
final_counts_df.to_csv(output_file, index=False)
