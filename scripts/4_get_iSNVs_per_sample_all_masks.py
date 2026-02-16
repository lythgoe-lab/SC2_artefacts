"""
This script processes the iSNV data for each sample across all sequencing sites and summarises the number of iSNVs per sample after applying various masking conditions and using various 
analysis MAF thresholds. It applies both static (no mask or the prevalent mask) and dynamic masks (all different combinations of MAF and sharing % thresholds) to the data, 
counts the iSNVs for each sample, and saves the results in a long format for downstream analysis. The output is saved in parquet format for efficient storage and retrieval.
"""
import os
import json
import site
import pandas as pd
import re


base_processed_dir = '../processed_data'
cov_folders = ["10x"]  # folder of filtered data meeting this coverage threshold in >50% of the genome
MASK_READ_DEPTH = 1000  # read depth threshold per position when generating mask sets (only positions reaching this depth considered as iSNVs and potentially masked)
COUNT_READ_DEPTHS = [1000]  # read depth threshold per position when analysing iSNVs and counting them (before and after masking)
analysis_maf_thresholds = list(range(2, 21))  # 2% to 20%

site_to_folder_map = {
    "NORT_ARTIC_3": "NORT_ARTIC_3_ILLUMINA",
    "NORT_ARTIC_4": "NORT_ARTIC_4_ILLUMINA",
    "NORT_ARTIC_4.1": "NORT_ARTIC_4.1_ILLUMINA",
    "NORW_ARTIC_4.1": "NORW_ARTIC_4.1_ILLUMINA",
    "OXON_VeSeq": "OXON_VeSeq_ILLUMINA",
    "PHEC_ARTIC_3": "PHEC_ARTIC_3_ILLUMINA",
    "SANG_ARTIC_3": "SANG_ARTIC_3_ILLUMINA",
    "SANG_ARTIC_4.1": "SANG_ARTIC_4.1_ILLUMINA"
}

# Static masks
static_mask_conditions = ["no_mask", "positions_in_over_20pct_masks"]


# Helper functions

def load_combined_mask(mask_folder_path, mask_name=None):
    combined_mask_file = os.path.join(mask_folder_path, "all_masks.csv")
    if os.path.exists(combined_mask_file):
        try:
            df = pd.read_csv(combined_mask_file)
            if mask_name is not None:
                df = df[df["mask_file"] == mask_name]
            return df
        except Exception as e:
            print(f"[Warning] Could not read combined mask file {combined_mask_file}: {e}")
            return None
    else:
        print(f"[Warning] all_masks.csv not found in {mask_folder_path}")
        return None

def get_dynamic_mask_conditions(mask_folder_path):
    df = load_combined_mask(mask_folder_path)
    if df is None:
        return []

    mask_names = df["mask_file"].dropna().unique().tolist()
    return [
        name for name in mask_names
        if re.match(r"\d+(?:\.\d+)?%shared_\d+%MAF", name)
    ]

def convert_freq(val):
    if isinstance(val, str) and '%' in val:
        try:
            return float(val.rstrip('%')) / 100.0
        except Exception:
            return pd.NA
    return val


# Process each site and apply masks

def process_site(site, mask_conditions, cov_folder, count_read_depth, mask_read_depth):
    folder = site_to_folder_map.get(site)
    if folder is None:
        print(f"[Warning] Folder for site {site} not found in mapping.")
        return None

    input_file_path = os.path.join(base_processed_dir, cov_folder, folder, "output_file.csv")
    if not os.path.exists(input_file_path):
        print(f"[Warning] Input file {input_file_path} not found for site {site}.")
        return None

    print(f"Processing site: {site}")
    data = pd.read_csv(input_file_path, low_memory=False)
    data = data.replace('N/A', pd.NA)
    data['minor_allele1_freq_no_gaps'] = data['minor_allele1_freq_no_gaps'].apply(convert_freq)
    data = data[(data['Position'] > 264) & (data['Position'] < 29675) & (data['major_allele'] != 'gap')]
    data = data[data['read_depth_no_gaps'] >= count_read_depth]

    sample_ids = pd.DataFrame({'ID': data['ID'].unique()})
    condition_counts = {}

    mask_folder_path = os.path.join(
        base_processed_dir, cov_folder, folder,
        f"mask_files_depth_{mask_read_depth}x")

    for condition in mask_conditions:
        print(f"Applying mask: {condition}")
        if condition == "no_mask":
            condition_data = data.copy()
        else:
            if condition.startswith("positions_in_over_"):
                mask_csv_file = os.path.join(mask_folder_path, f"{condition}.csv")
                if os.path.exists(mask_csv_file):
                    try:
                        df_mask = pd.read_csv(mask_csv_file)
                        mask_positions = set(df_mask["Position"].dropna().astype(int))
                        condition_data = data[~data['Position'].isin(mask_positions)]
                    except Exception as e:
                        print(f"[Error] Reading mask file {mask_csv_file}: {e}")
                        condition_data = None
                else:
                    print(f"[Warning] Mask file {mask_csv_file} not found.")
                    condition_data = None       
            else:
                mask_df = load_combined_mask(mask_folder_path, mask_name=condition)
                if mask_df is None or mask_df.empty:
                    print(f"[Warning] Mask '{condition}' not found for site {site}. Columns will be NA.")
                    condition_data = None
                else:
                    try:
                        mask_positions = set(json.loads(mask_df.iloc[0]["positions"]))
                        condition_data = data[~data['Position'].isin(mask_positions)]
                    except Exception as e:
                        print(f"[Error] Parsing mask positions for '{condition}' in site {site}: {e}")
                        condition_data = None
        
        counts_df = sample_ids.copy()
        for maf in analysis_maf_thresholds:
            col_name = f"iSNVs_{condition}_{maf}%analysis_MAF"
            if condition_data is None:
                counts_df[col_name] = pd.NA
            else:
                maf_threshold = maf / 100.0
                filtered = condition_data[condition_data['minor_allele1_freq_no_gaps'] >= maf_threshold]
                grp = filtered.groupby('ID').size().reset_index(name=col_name)
                counts_df = counts_df.merge(grp, on='ID', how='left')
        
        if condition_data is not None:
            count_cols = [col for col in counts_df.columns if col.startswith("iSNVs_")]
            counts_df[count_cols] = counts_df[count_cols].fillna(0).astype(int)
        
        condition_counts[condition] = counts_df

    long_rows = []
    for cond, df_cond in condition_counts.items():
        id_col = df_cond['ID']
        for maf in analysis_maf_thresholds:
            col_name = f"iSNVs_{cond}_{maf}%analysis_MAF"
            counts_col = df_cond[col_name] if col_name in df_cond.columns else pd.NA
            long_rows.append(pd.DataFrame({
                'ID': id_col,
                'Site': site,
                'Mask_Condition': cond,
                'MAF_Threshold_Percent': maf,
                'iSNV_Count': counts_col
            }))
    long_df = pd.concat(long_rows, ignore_index=True)
    return long_df



def main():
    chunk_size = 10 ** 6
    for cov_folder in cov_folders:
        for count_read_depth in COUNT_READ_DEPTHS:
            print(f"\nProcessing coverage folder: {cov_folder} at read depth: {count_read_depth}x")

            # get mask conditions once
            first_site_folder = next(iter(site_to_folder_map.values()))
            mask_base = os.path.join(base_processed_dir, cov_folder, first_site_folder, f"mask_files_depth_{MASK_READ_DEPTH}x")
            dynamic_mask_conditions = get_dynamic_mask_conditions(mask_base)
            mask_conditions = static_mask_conditions + dynamic_mask_conditions
            print(f"\nTotal mask conditions: {len(mask_conditions)}")

            output_path = os.path.join(base_processed_dir, cov_folder, f"total_isnvs_per_sample_and_maf_all_masks_depth_{MASK_READ_DEPTH}x")
            os.makedirs(output_path, exist_ok=True)
            
            part = 0
            
            for site in site_to_folder_map:
                res = process_site(site, mask_conditions, cov_folder, count_read_depth, MASK_READ_DEPTH)
                if res is None or res.empty:
                    continue
                # write the site's result in chunks
                total = len(res)
                for i in range(0, total, chunk_size):
                    chunk = res.iloc[i:i + chunk_size]
                    path = os.path.join(output_path, f"total_iSNVs_per_sample_depth_masked_part_{part}.parquet")
                    chunk.to_parquet(path, compression='gzip')
                    print(f' Wrote rows {i}-{min(i + chunk_size, total) - 1} to {path}')
                    part += 1
                # free memory
                del res
            
            if part == 0:
                print(f"No data processed for coverage folder {cov_folder} at read depth {count_read_depth}x.")
                continue

            print(f"{part} files written to {output_path} for coverage folder {cov_folder} at read depth {count_read_depth}x.")

if __name__ == "__main__":
    main()
