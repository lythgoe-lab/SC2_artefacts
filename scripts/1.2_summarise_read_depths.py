"""
This script processes Parquet files containing combined BaseFreq data for all samples at each sequencing centre and coverage threshold. 
It computes mean read depths and counts of samples meeting specific depth thresholds for each genomic position.
The output is a CSV file summarising this read depth information per position for each sequencing centre and coverage threshold.
"""

import os
import dask.dataframe as dd
import pandas as pd

sites_to_run = [
    "BRBR_ARTIC_4.1_ILLUMINA", "LSPA_ARTIC_4.1_ILLUMINA", "MILK_ARTIC_3_ILLUMINA", 
    "NORT_ARTIC_3_ILLUMINA", "NORT_ARTIC_4_ILLUMINA", "NORT_ARTIC_4.1_ILLUMINA",
    "NORW_ARTIC_4.1_ILLUMINA", "OXON_VeSeq_ILLUMINA", "PHEC_ARTIC_3_ILLUMINA", 
    "QEUH_ARTIC_3_ILLUMINA", "QEUH_ARTIC_4.1_ILLUMINA"
]
threshold_folders = ["1x", "10x", "100x", "1000x"]

def process_site_with_dask(input_folder, site):
    """
    Reads all Parquet files in input_folder into a Dask DataFrame,
    renames the first column to "Position", computes depth metrics and threshold counts,
    groups by Position, and returns a Dask DataFrame with the aggregated results
    """
    ddf = dd.read_parquet(input_folder, engine="pyarrow")
    
    ddf = ddf.rename(columns={ddf.columns[0]: "Position"})
 
    ddf["total_depth"] = ddf[["A count", "C count", "G count", "T count", "gap count"]].sum(axis=1)
    ddf["no_gap_depth"] = ddf[["A count", "C count", "G count", "T count"]].sum(axis=1)
    
    ddf["Site"] = site
    ddf["row_count"] = 1
    
    # Create threshold count columns
    ddf["count_10_total"] = (ddf["total_depth"] >= 10).astype(int)
    ddf["count_10_no_gap"] = (ddf["no_gap_depth"] >= 10).astype(int)
    ddf["count_100_total"] = (ddf["total_depth"] >= 100).astype(int)
    ddf["count_100_no_gap"] = (ddf["no_gap_depth"] >= 100).astype(int)
    ddf["count_1000_total"] = (ddf["total_depth"] >= 1000).astype(int)
    ddf["count_1000_no_gap"] = (ddf["no_gap_depth"] >= 1000).astype(int)

    # Group by Position and sum up all values
    grouped = ddf.groupby("Position").agg({
        "total_depth": "sum",
        "no_gap_depth": "sum",
        "row_count": "sum",
        "count_10_total": "sum",
        "count_10_no_gap": "sum",
        "count_100_total": "sum",
        "count_100_no_gap": "sum",
        "count_1000_total": "sum",
        "count_1000_no_gap": "sum"
    }).reset_index()

    grouped["mean_read_depth"] = grouped["total_depth"] / grouped["row_count"]
    grouped["mean_read_depth_no_gaps"] = grouped["no_gap_depth"] / grouped["row_count"]
    
    grouped["Site"] = site
    
    # Select and rename columns to match desired output
    result = grouped[["Site", "Position", "mean_read_depth", "mean_read_depth_no_gaps",
                      "count_10_total", "count_10_no_gap", "count_100_total", "count_100_no_gap",
                      "count_1000_total", "count_1000_no_gap"]]
    
    result = result.rename(columns={
        "count_10_total": "samples_depth_10",
        "count_10_no_gap": "samples_depth_10_no_gaps",
        "count_100_total": "samples_depth_100",
        "count_100_no_gap": "samples_depth_100_no_gaps",
        "count_1000_total": "samples_depth_1000",
        "count_1000_no_gap": "samples_depth_1000_no_gaps"
    })
    
    return result


def main():
    base_dir = "."
    
    for threshold in threshold_folders:
        print(f"Processing threshold folder: {threshold}")
        site_ddfs = []
        for site in sites_to_run:
            site_folder = os.path.join(base_dir, "combined_basefreqs_by_site", threshold, site)
            if not os.path.exists(site_folder):
                print(f"Warning: {site_folder} not found. Skipping {site}.")
                continue
            # Simplify site name (remove "_ILLUMINA")
            simple_site = site.replace("_ILLUMINA", "")
            print(f"Processing site: {site} as {simple_site}")
            try:
                site_result = process_site_with_dask(site_folder, simple_site)
                site_ddfs.append(site_result)
            except Exception as e:
                print(f"Error processing site {site}: {e}")
        
        if not site_ddfs:
            print(f"No data for threshold {threshold}.")
            continue
        
        # Combine all site Dask DataFrames and compute (trigger parallel processing)
        combined = dd.concat(site_ddfs)
        combined = combined.compute()
        

        # Group sites into SANG 
        SANG_ARTIC_3_sites = ["MILK_ARTIC_3", "QEUH_ARTIC_3"]
        SANG_ARTIC_3 = combined[combined["Site"].isin(SANG_ARTIC_3_sites)]
        if not SANG_ARTIC_3.empty:
            SANG_ARTIC_3 = SANG_ARTIC_3.groupby("Position", as_index=False).agg({
                "mean_read_depth": "mean",
                "mean_read_depth_no_gaps": "mean",
                "samples_depth_10": "sum",
                "samples_depth_10_no_gaps": "sum",
                "samples_depth_100": "sum",
                "samples_depth_100_no_gaps": "sum",
                "samples_depth_1000": "sum",
                "samples_depth_1000_no_gaps": "sum"
            })
            SANG_ARTIC_3["Site"] = "SANG_ARTIC_3"
        
        SANG_ARTIC_4_1_sites = ["BRBR_ARTIC_4.1", "LSPA_ARTIC_4.1", "QEUH_ARTIC_4.1"]
        SANG_ARTIC_4_1 = combined[combined["Site"].isin(SANG_ARTIC_4_1_sites)]
        if not SANG_ARTIC_4_1.empty:
            SANG_ARTIC_4_1 = SANG_ARTIC_4_1.groupby("Position", as_index=False).agg({
                "mean_read_depth": "mean",
                "mean_read_depth_no_gaps": "mean",
                "samples_depth_10": "sum",
                "samples_depth_10_no_gaps": "sum",
                "samples_depth_100": "sum",
                "samples_depth_100_no_gaps": "sum",
                "samples_depth_1000": "sum",
                "samples_depth_1000_no_gaps": "sum"
            })
            SANG_ARTIC_4_1["Site"] = "SANG_ARTIC_4.1"
        else:
            SANG_ARTIC_4_1 = pd.DataFrame()
        
        # Combine individual site summaries and SANG summaries
        final_summary = pd.concat([combined, SANG_ARTIC_3, SANG_ARTIC_4_1], ignore_index=True)
        final_summary.sort_values(["Site", "Position"], inplace=True)
        
        output_folder = os.path.join(base_dir, "processed_data", threshold)
        os.makedirs(output_folder, exist_ok=True)
        output_file = os.path.join(output_folder, "read_depth_summary.csv")
        final_summary.to_csv(output_file, index=False)
        print(f"Final summarising file saved to: {output_file}")

if __name__ == "__main__":
    main()
