"""
This script combines all the mask files in each "mask_files_depth_" subfolder into a single CSV file named "all_masks.csv".
The combined file will have two columns: "mask_file" (the name of the original mask file without the .csv extension) and "positions" (a JSON string containing 
the list of unique positions from that mask file). 
"""

import os
import json
import pandas as pd

base_dir = '../processed_data/'

coverage_folders = ["1x", "10x", "100x", "1000x"]

def parse_mask_filename(mask_name):
    if mask_name.endswith('.csv'):
        mask_name = mask_name[:-4]
    try:
        parts = mask_name.split("shared_")
        if len(parts) != 2:
            raise ValueError("Filename does not contain 'shared_'")
        shared_part, maf_part = parts
        # Remove any "%" and extra spaces
        shared = float(shared_part.replace("%", "").strip())
        maf = float(maf_part.replace("%MAF", "").strip())
        return maf, shared
    except Exception as e:
        print(f"Error parsing mask filename '{mask_name}': {e}")
        return None, None

# Walk through each coverage folder and each site folder within it
for cov in coverage_folders:
    cov_path = os.path.join(base_dir, cov)
    site_folders = [d for d in os.listdir(cov_path) if os.path.isdir(os.path.join(cov_path, d))]
    for site in site_folders:
        site_path = os.path.join(cov_path, site)
        # Look for subfolders that start with "mask_files_depth_"
        for subfolder in os.listdir(site_path):
            if subfolder.startswith("mask_files_depth_"):
                mask_folder = os.path.join(site_path, subfolder)
                if not os.path.isdir(mask_folder):
                    continue
                # List all CSV mask files that contain "shared_" in the filename
                mask_files = [f for f in os.listdir(mask_folder) if f.endswith('.csv') and "shared_" in f]
                combined_list = []
                for mask_file in mask_files:
                    # Parse the filename to extract numeric MAF and Shared values
                    maf_val, shared_val = parse_mask_filename(mask_file)
                    if maf_val is None or shared_val is None:
                        print(f"Skipping file '{mask_file}' because it does not match the expected pattern.")
                        continue
                    mask_path = os.path.join(mask_folder, mask_file)
                    try:
                        # Read the mask file (assume one column of positions without header)
                        df_mask = pd.read_csv(mask_path, header=None, names=["positions"])
                        # Extract unique positions as a sorted list
                        positions = sorted(df_mask["positions"].dropna().unique().tolist())
                        combined_list.append({
                            "mask_file": mask_file[:-4],  # remove the ".csv"
                            "positions": json.dumps(positions),
                            "MAF": maf_val,
                            "Shared": shared_val
                        })
                    except Exception as e:
                        print(f"Error reading {mask_path}: {e}")
                if combined_list:
                    combined_df = pd.DataFrame(combined_list)
                    # Sort by MAF first, then by Shared percentage
                    combined_df = combined_df.sort_values(by=["MAF", "Shared"]).reset_index(drop=True)
                    # Keep only the desired columns: 'mask_file' and 'Position'
                    combined_df = combined_df[["mask_file", "positions"]]
                    
                    # Write the combined DataFrame to a new CSV file
                    output_path = os.path.join(mask_folder, "all_masks.csv")
                    combined_df.to_csv(output_path, index=False)
                    print(f"Combined mask file written to: {output_path}")
