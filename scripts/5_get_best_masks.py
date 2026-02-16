"""
This script compares the mean iSNV count per sample for each sequencing centre and protocol across different MAF thresholds to a reference dataset (OXON with no mask at 3% MAF threshold) 
to identify the best mask for each site that makes the mean iSNV count per sample match the reference most closely. 
It then extracts the positions for these best masks and the prevalent masks (positions in over 20% of masks) from the respective mask folders and compiles this information into a summary CSV file.
"""

import os
import json
import pandas as pd
import numpy as np
import dask.dataframe as dd
from dask.diagnostics import ProgressBar

base_processed_dir = '../processed_data'
cov_folder = '10x'
# path to parquet files containing iSNV counts per sample before and after masking at different MAF thresholds for all sites
parquet_path = os.path.join(base_processed_dir, cov_folder, 'total_isnvs_per_sample_and_maf_all_masks_depth_1000x/*.parquet')
read_depth = 1000

# reference dataset (here OXON with no mask analysed at a 3% MAF threshold) for identifying the best masks for each sequencing centre and protocol that most closely 
# matches the mean iSNV count per sample to the reference dataset.
ref_site = 'OXON_VeSeq'
ref_mask_name = 'no_mask'
ref_maf_threshold = 3

common_mask = 'positions_in_over_20pct_masks'
shared_prefix = "1.5%shared"

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

print(f"Loading data from {parquet_path}...")

# Load parquet data with Dask, selecting only relevant columns and categorising string columns for efficiency
ddf = dd.read_parquet(
    parquet_path,
    engine='pyarrow',
    columns=['Site', 'Mask_Condition', 'MAF_Threshold_Percent', 'iSNV_Count']
).categorize(columns=['Site', 'Mask_Condition'])

print(f"Filtering data for unmasked and {shared_prefix} masks...\n")
ddf = ddf[ddf['Mask_Condition'].isin(['no_mask']) | ddf['Mask_Condition'].str.startswith(shared_prefix)]

print("Computing mean iSNVs per sample for each site, mask condition, and MAF threshold...\n")
ProgressBar().register()
# Group by Site, Mask_Condition, and MAF_Threshold_Percent to calculate mean iSNV count per sample, then pivot to have MAF thresholds as index and (Site, Mask_Condition) as columns
mean_ddf = (
    ddf.groupby(['Site', 'Mask_Condition', 'MAF_Threshold_Percent'])
       .iSNV_Count.mean()
       .persist()
)
mean_df = mean_ddf.compute().reset_index()
pivoted = mean_df.pivot_table(
    index='MAF_Threshold_Percent',
    columns=['Site', 'Mask_Condition'],
    values='iSNV_Count'
)

# Get OXON reference data at 3% MAF threshold
print(f"Retrieving reference data for {ref_site} at {ref_mask_name} {ref_maf_threshold}%...\n")
ref_series = pivoted.get((ref_site, ref_mask_name))
if ref_series is None or ref_maf_threshold not in ref_series:
    raise ValueError(f"Reference data not found for {ref_site} at {ref_mask_name} {ref_maf_threshold}%")
ref_y = ref_series.loc[ref_maf_threshold]

# Prepare to find best masks MAF threshold for each site
print("Identifying best masks for each site...")
best_masks = {}

for site in sites:
    try:
        y_unmasked = pivoted[(site, 'no_mask')]
        diffs = (y_unmasked - ref_y).abs()
        best_maf = int(diffs.idxmin())
        closest_val = y_unmasked[best_maf]
        best_mask_name = f'1.5%shared_{best_maf}%MAF'
        best_masks[site] = best_mask_name
        print(f"{site}: best mask is {best_mask_name} - unmasked data closest to {ref_site}'s {ref_mask_name} masked data at MAF ≥ {ref_maf_threshold}% ({ref_y:.2f} average iSNVs) "
              f"at {best_maf}%MAF ({closest_val:.2f} iSNVs per sample)")
    except Exception as e:
        print(f"[Warning] Could not compute best mask for {site}: {e}")

# Function to get mask positions
def load_mask_positions(mask_folder, mask_name):
    if mask_name == 'positions_in_over_20pct_masks':
        common_file = os.path.join(mask_folder, f"{mask_name}.csv")
        if not os.path.exists(common_file):
            print(f"[Warning] {mask_name}.csv not found in {mask_folder}")
            return []
        try:
            df = pd.read_csv(common_file)
            return df['Position'].dropna().astype(int).tolist()
        except Exception as e:
            print(f"[Warning] Error reading {common_file}: {e}")
            return []
    else:
        csv_path = os.path.join(mask_folder, 'all_masks.csv')
        if not os.path.exists(csv_path):
            print(f"[Warning] Missing all_masks.csv in {mask_folder}")
            return []
        try:
            df = pd.read_csv(csv_path, converters={"positions": json.loads})
            row = df[df["mask_file"] == mask_name]
            if row.empty:
                print(f"[Warning] Mask '{mask_name}' not found in {csv_path}")
                return []
            return [int(p) for p in row["positions"].iloc[0]]
        except Exception as e:
            print(f"[Warning] Error reading {csv_path}: {e}")
            return []

# Extract positions from mask folders
print("\nExtracting positions from mask folders...")
rows = []

for site, folder in site_to_folder_map.items():
    mask_folder = os.path.join(base_processed_dir, cov_folder, folder, f"mask_files_depth_{read_depth}x")

    # Add best mask
    best = best_masks.get(site)
    if best:
        positions = load_mask_positions(mask_folder, best)
        rows.append({
            "Centre_protocol": site,
            "mask": best,
            "positions": json.dumps(positions)
        })

    # Add common mask
    positions_common = load_mask_positions(mask_folder, common_mask)
    rows.append({
        "Centre_protocol": site,
        "mask": common_mask,
        "positions": json.dumps(positions_common)
    })

# Write final mask summary file
summary_df = pd.DataFrame(rows)
output_csv = os.path.join(base_processed_dir, cov_folder, "final_masks_all_centres.csv")
summary_df.to_csv(output_csv, index=False)
print(f"Wrote {output_csv} with {len(summary_df)} rows.")
