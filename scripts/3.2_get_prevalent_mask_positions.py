"""
This script summarises the most prevalently masked positions across all the different mask files (x%MAF and y%shared) for each sequencing site. 
It identifies positions that are masked in more than a specified percentage of all the mask files for that site, and writes these positions to a new CSV file in the respective site's mask folder. 
This is used as an alternative masking strategy and shown in the supplementary material of the paper.
"""

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import matplotlib.lines as mlines

base_processed_dir = "../processed_data/"

site_to_folder_map = {
    "NORT_ARTIC_3": "NORT_ARTIC_3_ILLUMINA",
    "NORT_ARTIC_4": "NORT_ARTIC_4_ILLUMINA",
    "NORT_ARTIC_4.1": "NORT_ARTIC_4.1_ILLUMINA",
    "NORW_ARTIC_4.1": "NORW_ARTIC_4.1_ILLUMINA",
    "OXON_VeSeq": "OXON_VeSeq_ILLUMINA",
    "PHEC_ARTIC_3": "PHEC_ARTIC_3_ILLUMINA",
    "SANG_ARTIC_3": "SANG_ARTIC_3_ILLUMINA",
    "SANG_ARTIC_4.1": "SANG_ARTIC_4.1_ILLUMINA",
}

sites = [
    "OXON_VeSeq",
    "PHEC_ARTIC_3",
    "NORT_ARTIC_3",
    "NORT_ARTIC_4",
    "NORT_ARTIC_4.1",
    "NORW_ARTIC_4.1",
    "SANG_ARTIC_3",
    "SANG_ARTIC_4.1"
]

# choose coverage folder and mask depth folder
cov_folder  = "10x"
mask_folder = "mask_files_depth_1000x"

# Threshold for minimum percentage of mask files containing position
threshold = 0.2   # 20%


# Regex to match mask filenames like "1.5%shared_3%MAF.csv"
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

# Collect all mask records
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

df = pd.DataFrame(records)

# Compute masks count per sequencing site for normalization
masks_per_site = df.groupby("site")["mask"].nunique().to_dict()

# Build per-sequencing site position frequencies
site_position_freq = {}
for site in sites:
    df_site = df[df["site"] == site]
    total_masks = masks_per_site.get(site, 0)
    pos_counts = defaultdict(int)
    for pos_list in df_site["masked_positions"]:
        for pos in pos_list:
            pos_counts[pos] += 1

    sorted_pos = sorted(pos_counts.keys())
    proportions = [pos_counts[p] / total_masks for p in sorted_pos]
    site_position_freq[site] = (sorted_pos, proportions)

# Write positions above threshold to CSV in each site's mask folder
for site in sites:
    positions, proportions = site_position_freq.get(site, ([], []))
    above_positions = [p for p, prop in zip(positions, proportions) if prop > threshold]

    folder = site_to_folder_map[site]
    out_dir = os.path.join(base_processed_dir, cov_folder, folder, mask_folder)
    os.makedirs(out_dir, exist_ok=True)

    out_file = os.path.join(out_dir,
        f"positions_in_over_{int(threshold*100)}pct_masks.csv")
    pd.DataFrame({"Position": above_positions}).to_csv(out_file, index=False)
    print(f"Wrote {len(above_positions)} positions to {out_file}")