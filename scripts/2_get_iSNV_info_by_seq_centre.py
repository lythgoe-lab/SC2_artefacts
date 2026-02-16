"""
This script processes the summarised iSNV data for each sequencing site and coverage folder (output.csv from 1_process_basefreq_files.py) to create two summary tables:
- samples_per_iSNV_position.csv: For each position, the number of samples with iSNVs at different MAF and depth thresholds.
- total_iSNVs_per_sample.csv: For each sample, the total number of iSNVs at different MAF and depth thresholds.

The script iterates through each sequencing site and coverage folder, applies the specified MAF and depth thresholds, 
and saves the resulting summary tables in the corresponding sequencing site folder for each coverage level. 

Only positions within the coding region of the SARS-CoV-2 genome are included, and positions with a major allele of 'gap' are excluded. Total read depths and minor allele frequencies
are also excluding gaps.

The output files are used in subsequent steps to analyse iSNV distributions and to identify highly shared iSNV positions for masking.
"""


import pandas as pd
import os

sites_to_run = ["NORT_ARTIC_3_ILLUMINA", "NORT_ARTIC_4_ILLUMINA", "NORT_ARTIC_4.1_ILLUMINA",
                "NORW_ARTIC_4.1_ILLUMINA", "OXON_VeSeq_ILLUMINA", "PHEC_ARTIC_3_ILLUMINA",
                "SANG_ARTIC_3_ILLUMINA", "SANG_ARTIC_4.1_ILLUMINA"]

coverage_folders = ["1x", "10x", "100x", "1000x"]

base_dir = '../'

depth_thresholds = [10, 100, 1000]
maf_thresholds = [x / 100.0 for x in range(2, 21)]  # From 2% to 20%

def main():
    for coverage_folder in coverage_folders:
        print(f"\n{'#'*60}\nProcessing coverage folder: {coverage_folder}\n{'#'*60}\n")

        for site in sites_to_run:
            print(f"Processing site: {site}")

            # Make paths for input and output files for this site
            input_file_path = os.path.join(base_dir, "processed_data", coverage_folder, site, "output_file.csv")
            if not os.path.exists(input_file_path):
                print(f"ERROR: {input_file_path} not found. Skipping...\n")
                continue
            output_folder = os.path.join(base_dir, "processed_data", coverage_folder, site)
            os.makedirs(output_folder, exist_ok=True)

            # Read input CSV for this site
            print(f"Reading {input_file_path} ...")
            data = pd.read_csv(input_file_path, low_memory=False)

            # Convert MAF percentages to floats and exclude specific positions
            data = data.replace('N/A', pd.NA)
            data['minor_allele1_freq_no_gaps'] = data['minor_allele1_freq_no_gaps'].apply(
                lambda x: float(x.rstrip('%')) / 100.0 if isinstance(x, str) and '%' in x else x)
            data = data[(data['Position'] > 264) & (data['Position'] < 29675) & (data['major_allele'] != 'gap')]

            samples_per_iSNV_position_list = []
            
            # Create table to hold total iSNVs per sample
            all_ids = data['ID'].unique()
            total_iSNVs_per_sample = pd.DataFrame({'ID': all_ids})

            # Process data for each read depth threshold to make summaries of total iSNVs per sample and samples per iSNV position
            for depth_threshold in depth_thresholds:
                # Filter the data by read depth
                filtered_data = data[data['read_depth_no_gaps'] >= depth_threshold]
                # Build a set of sample IDs that have sufficient coverage at this threshold
                samples_with_coverage = set(filtered_data['ID'].unique())

                # DataFrame for per-position counts of samples with iSNVs at this depth threshold
                samples_per_iSNV_position = pd.DataFrame()

                # Process each MAF threshold
                for maf_threshold in maf_thresholds:
                    maf_filtered_data = filtered_data[filtered_data['minor_allele1_freq_no_gaps'] >= maf_threshold]
                    
                    # For per-position counts: count unique sample IDs that pass MAF filtering
                    position_counts = maf_filtered_data.groupby('Position')['ID'].nunique().reset_index(
                        name=f'Samples_MAF_{int(maf_threshold * 100)}%_depth_{depth_threshold}x')

                    # For per-sample iSNV counts: group by sample ID
                    isnv_count = maf_filtered_data.groupby('ID').size()
                    
                    # New column name for this threshold combination
                    col_name = f'iSNVs_{int(maf_threshold * 100)}%_depth_{depth_threshold}x'
                    
                    # only assign 0 if the sample has no iSNVs but have sufficient coverage, otherwise leave as NA if too low coverage
                    def get_count(sample_id):
                        if sample_id in samples_with_coverage:
                            return isnv_count.get(sample_id, 0)
                        else:
                            return pd.NA
                    total_iSNVs_per_sample[col_name] = total_iSNVs_per_sample['ID'].apply(get_count)
                    
                    # For the long format per-position counts, merge the counts for this threshold with the existing DataFrame
                    if samples_per_iSNV_position.empty:
                        samples_per_iSNV_position = position_counts
                    else:
                        samples_per_iSNV_position = pd.merge(samples_per_iSNV_position, position_counts, on='Position', how='outer')

                # Append the long format results for this depth threshold to the list
                samples_per_iSNV_position_list.append(samples_per_iSNV_position)

            # Merge all per-position DataFrames on 'Position'
            final_samples_per_iSNV_position = pd.DataFrame()
            for df_pos in samples_per_iSNV_position_list:
                if final_samples_per_iSNV_position.empty:
                    final_samples_per_iSNV_position = df_pos
                else:
                    final_samples_per_iSNV_position = pd.merge(final_samples_per_iSNV_position, df_pos, on='Position', how='outer')
            # Fill NaN with 0
            final_samples_per_iSNV_position.fillna(0, inplace=True)

            # Convert all iSNV count columns to integers (Int64 type to convert NaNs to pd.NA)
            isnv_columns = [col for col in total_iSNVs_per_sample.columns if col.startswith('iSNVs_')]
            total_iSNVs_per_sample[isnv_columns] = total_iSNVs_per_sample[isnv_columns].astype('Int64')

            # Save results to CSV
            final_samples_per_iSNV_position_path = os.path.join(output_folder, "samples_per_iSNV_position.csv")
            total_iSNVs_per_sample_path = os.path.join(output_folder, "total_iSNVs_per_sample.csv")

            final_samples_per_iSNV_position.to_csv(final_samples_per_iSNV_position_path, index=False)
            total_iSNVs_per_sample.to_csv(total_iSNVs_per_sample_path, index=False)

            print(f"Saved:\n   {final_samples_per_iSNV_position_path}\n   {total_iSNVs_per_sample_path}\n")

    print("All done!")

if __name__ == "__main__":
    main()
