"""
This script counts the total number of sample files (BaseFreq CSV files) for each sequencing centre and coverage threshold.
It outputs a CSV file summarising the total sample counts per site and coverage threshold, which can be used for downstream analyses 
such as calculating the percentage of samples with iSNVs at each position under different thresholds.
"""


import os
import glob
import csv

base_dir = "./basefreqs_by_site_filtered"

output_csv = "samples_per_site_depth_summary.csv"


results = []

# Iterate over each threshold directory in the base directory
for threshold in os.listdir(base_dir):
    threshold_path = os.path.join(base_dir, threshold)
    if os.path.isdir(threshold_path):
        # Iterate over each sequencing site directory inside the threshold folder
        for site in os.listdir(threshold_path):
            site_path = os.path.join(threshold_path, site)
            if os.path.isdir(site_path):
                # Count files matching the pattern "*_BaseFreqs.csv"
                file_pattern = os.path.join(site_path, "*_BaseFreqs.csv")
                file_list = glob.glob(file_pattern)
                count = len(file_list)
                results.append([site, threshold, count])

# Sort the results (by site then threshold)
results.sort()

with open(output_csv, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Site", "Cov_Threshold", "Total_Samples"])
    writer.writerows(results)

print(f"Summary saved to {output_csv}")
