"""
This script identifies “highly shared” iSNV positions and generates candidate masking schemes by varying minor allele frequency (MAF) and sample-sharing thresholds across samples within 
the same sequencing centre and protocol group. It processes iSNV calls and read-depth summaries for each sequencing site and coverage threshold to produce mask files for positions exceeding
the specified criteria.

For each combination of coverage threshold, depth requirement, MAF threshold, and sharing-percentage increment, the script determines the number of samples meeting the depth criterion
and identifies positions where the count of samples with iSNVs surpasses the corresponding sharing threshold. The resulting positions are written to CSV files organised by coverage threshold,
sequencing site, depth threshold, and MAF + sharing-percentage combination (e.g. 0.5%shared_2%MAF.csv within a mask_files_depth_10x directory).
"""

import pandas as pd
import os

#  sequencing sites and protocol folder names
sites_to_run = ["BRBR_ARTIC_4.1_ILLUMINA", "LSPA_ARTIC_4.1_ILLUMINA", "MILK_ARTIC_3_ILLUMINA",
                "NORT_ARTIC_3_ILLUMINA", "NORT_ARTIC_4_ILLUMINA", "NORT_ARTIC_4.1_ILLUMINA",
                "NORW_ARTIC_4.1_ILLUMINA", "OXON_VeSeq_ILLUMINA", "PHEC_ARTIC_3_ILLUMINA",
                "QEUH_ARTIC_3_ILLUMINA", "QEUH_ARTIC_4.1_ILLUMINA", 
                "SANG_ARTIC_3_ILLUMINA", "SANG_ARTIC_4.1_ILLUMINA"]


base_dir = '../'

# coverage folders to process (min coverage across 50% of the genome for included samples)
cov_thresholds = [1, 10, 100, 1000]

# depth thresholds to include position in iSNV counts or as denominator for sharing percentage
depth_thresholds = [10, 100, 1000]

# Masking MAF thresholds (2% to 20%)
masking_maf_thresholds = [i / 100 for i in range(2, 21)]
# Sharing percentages (from 0.5% to 20% as fractions)
sharing_percentages = [i / 200 for i in range(1, 41)]

def main():
    for site in sites_to_run:
        print("#" * 60)
        print(f"Processing site: {site}\n")
        # Remove the '_ILLUMINA' suffix from the site name for matching
        site_clean = site.replace("_ILLUMINA", "")

        # Loop over each coverage threshold folder
        for cov in cov_thresholds:
            cov_folder = f"{cov}x"
            print(f"Processing coverage threshold: {cov_folder}")

            isnv_results_path = os.path.join(base_dir, "processed_data", cov_folder, site, "samples_per_iSNV_position.csv")
            if not os.path.exists(isnv_results_path):
                print(f"iSNV results file not found for site {site} at coverage {cov_folder}: {isnv_results_path}")
                continue

            isnv_results = pd.read_csv(isnv_results_path, low_memory=False)
            expected_format_columns = {col: col.replace('.0', '') for col in isnv_results.columns}
            isnv_results.rename(columns=expected_format_columns, inplace=True)

            read_depth_summary_path = os.path.join(base_dir, "processed_data", cov_folder, "read_depth_summary.csv")
            if not os.path.exists(read_depth_summary_path):
                print(f"Read depth summary file not found for coverage {cov_folder}: {read_depth_summary_path}")
                continue

            depth_summary_df = pd.read_csv(read_depth_summary_path, low_memory=False)
            if "Site" in depth_summary_df.columns:
                depth_summary_df = depth_summary_df[depth_summary_df["Site"] == site_clean]

            # Process each depth threshold
            for depth in depth_thresholds:
                print(f"Processing depth threshold: {depth}x under coverage {cov_folder}")
                # Define the denominator column using the no-gaps column from the read depth summary
                denominator_col = f'samples_depth_{depth}_no_gaps'
                if denominator_col not in depth_summary_df.columns:
                    print(f"Column {denominator_col} not found in read depth summary for coverage {cov_folder}. Skipping depth {depth}x.")
                    continue

                # Merge iSNV results with the depth summary on "Position"
                merged_df = pd.merge(isnv_results, depth_summary_df, on="Position", how="inner")
                if merged_df.empty:
                    print(f"No matching positions found between iSNV results and read depth summary for site {site} at coverage {cov_folder} for depth {depth}x. Skipping.")
                    continue

                # Create output directory within the current coverage folder, inside the site folder
                output_dir = os.path.join(base_dir, "processed_data", cov_folder, site, f"mask_files_depth_{depth}x")
                os.makedirs(output_dir, exist_ok=True)

                # Iterate over each masking MAF threshold and sharing percentage increment
                for masking_maf_threshold in masking_maf_thresholds:
                    # Identify the appropriate column in the iSNV results for this MAF threshold and depth
                    isnv_col = f'Samples_MAF_{int(masking_maf_threshold * 100)}%_depth_{depth}x'
                    if isnv_col not in merged_df.columns:
                        print(f"Column {isnv_col} not found in iSNV results for site {site} at coverage {cov_folder}. Skipping this MAF threshold")
                        continue

                    any_positions_masked = False  # Flag to track if any positions were masked for this MAF threshold

                    # Loop over each sharing percentage increment
                    for increment in sharing_percentages:
                        # Compute per-position masking threshold = (samples meeting depth) * (sharing percentage)
                        threshold_series = (merged_df[denominator_col] * increment).astype(int)

                        # Identify positions where the number of samples with iSNVs exceeds the threshold
                        mask_sites = merged_df[merged_df[isnv_col] > threshold_series]
                        current_pos_count = mask_sites.shape[0]
                        formatted_percentage = f"{increment * 100:.1f}%"

                        # Define the output filename using the sharing percentage and MAF threshold
                        filename = f'{formatted_percentage}shared_{int(masking_maf_threshold * 100)}%MAF.csv'
                        output_path = os.path.join(output_dir, filename)

                        if current_pos_count > 0:
                            mask_sites[['Position']].to_csv(output_path, index=False, header=False)
                            print(f"Written {current_pos_count} positions to {output_path}")
                            any_positions_masked = True
                        else:
                            print(f"No positions to mask for {formatted_percentage} shared and MAF {int(masking_maf_threshold * 100)}% at depth {depth}x under coverage {cov_folder}")

                    if not any_positions_masked:
                        print(f"No positions were masked for MAF {int(masking_maf_threshold * 100)}% at depth {depth}x under coverage {cov_folder} with any sharing increment")

                print(f"Finished processing depth {depth}x for site {site} under coverage {cov_folder}\n")
            print(f"Finished processing coverage threshold {cov_folder} for site {site}\n")
        print(f"Finished processing site {site}\n")
    print("All done!")

if __name__ == "__main__":
    main()
