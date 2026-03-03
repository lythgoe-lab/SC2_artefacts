"""
This script processes ONS base frequency files and summarises all major/minor alleles for each sample per sequencing centre and protocol (site).

It reads all *_BaseFreqs.csv files in the specified input folders, calculates the major and minor alleles
(and their frequencies) at each position, and writes the results to output CSV files. 

Since these csv files can be very large, the script saves intermediate files after processing a set number of files
and then merges them at the end to avoid memory issues.
It processes whatever subset of sites is specified in the "sites_to_run" list and can this way be run in parallel on the for different sets of sites to save time.
Ideally, this script should be merged with the "2_compress_convert_variant_output_to_parquet.py" if rerun, to save space and time and writing directly to parquet instead of csv.

The script expects a "coverage" folder (e.g. "10x") containing subfolders for each site (e.g. "NORT_ARTIC_4.1_ILLUMINA"),
which in turn contain the respective *_BaseFreqs.csv files for each sample that meets the coverage threshold in half the genome.

It uses a minimum read depth threshold (here 10) to filter positions for major and minor allele calling and only outputs positions that meet this threshold.

For each position, the major and minor alleles (and their frequencies) are determined based on the base counts, and the total read depth excluding gaps. 
No minor allele frequency threshold is applied here; any detected minor allele with a frequency > 0 is reported.

If the major and minor alleles are tied exactly in counts (e.g. 50% A, 50% T), the reference base is chosen as the major allele if it is among the tied bases and a flag is set to indicate ambiguity.
If there is no exact tie but the top two alleles have frequencies close to 50% (between 45%-55%), the major allele is still called normally but a flag is set to indicate ambiguity.

Only the minor allele with the highest frequency is reported if multiple minor alleles are present.

"""

import pandas as pd
import os
import re
import glob
from Bio import SeqIO
from Bio.Seq import Seq
from datetime import datetime


# Select subset to run
sites_to_run = ["BRBR_ARTIC_4.1_ILLUMINA", "LSPA_ARTIC_4.1_ILLUMINA", "MILK_ARTIC_3_ILLUMINA", "QEUH_ARTIC_3_ILLUMINA",
                "NORW_ARTIC_Unknown_ILLUMINA", "OXON_VeSeq_ILLUMINA"]

#sites_to_run = ["NORT_ARTIC_3_ILLUMINA", "PHEC_ARTIC_3_ILLUMINA", "PHEC_ARTIC_Unknown_ILLUMINA"]

#sites_to_run = ["NORT_ARTIC_4.1_ILLUMINA"]

#sites_to_run = ["NORT_ARTIC_4_ILLUMINA"]

#sites_to_run = ["QEUH_ARTIC_4.1_ILLUMINA"]

#sites_to_run = ["BRBR_ARTIC_4.1_ILLUMINA", "LSPA_ARTIC_4.1_ILLUMINA", "MILK_ARTIC_3_ILLUMINA", 
#                "NORT_ARTIC_3_ILLUMINA", "NORT_ARTIC_4.1_ILLUMINA", "NORT_ARTIC_4_ILLUMINA",
#                "NORT_ARTIC_Unknown_ILLUMINA", "NORW_ARTIC_Unknown_ILLUMINA", "OXON_VeSeq_ILLUMINA",
#                "PHEC_ARTIC_3_ILLUMINA", "PHEC_ARTIC_Unknown_ILLUMINA", "QEUH_ARTIC_3_ILLUMINA",
#                "QEUH_ARTIC_4.1_ILLUMINA", "QEUH_ARTIC_Unknown_ILLUMINA", "QEUH_Unknown_ILLUMINA"]


# Min 50% coverage thresholds
threshold_folders = ["10x"]

MIN_READ_DEPTH = 10
intermediate_save_every = 500


# HELPER FUNCTIONS #

def preprocess_fasta(fasta_file):
    """Return dict mapping 1-based genome position to reference base from fasta file"""
    ref_seq_record = next(SeqIO.parse(fasta_file, "fasta"))
    seq = str(ref_seq_record.seq).upper()
    return {pos + 1: base for pos, base in enumerate(seq)}


def get_position(row):
    """Return the value from the first column starting with 'position in' """
    for col in row.index:
        if col.lower().startswith("position in"):
            return row[col]
    raise ValueError("No column starting with 'position in' found in the row")



def calculate_alleles(row, feature_sequences):
    """Determine major and minor alleles (and their frequencies) for a given row of base counts"""
    base_counts = {
        'A': row['A count'],
        'C': row['C count'],
        'G': row['G count'],
        'T': row['T count'],
        'gap': row['gap count']
    }

    total_count = sum(base_counts.values())
    total_count_no_gaps = total_count - base_counts['gap']

    if total_count == 0 or total_count_no_gaps == 0:
        # No usable coverage
        return 'no_coverage', None
    
    # Sort bases by count and find max frequency base (not considering gaps)
    nuc_counts = {b: base_counts[b] for b in ['A', 'C', 'G', 'T']}
    sorted_bases = sorted(nuc_counts.items(), key=lambda x: x[1], reverse=True)
    max_count = sorted_bases[0][1]

    # Detect tied major alleles
    tied_major_list = [allele for allele, count in sorted_bases if count == max_count]
    ambiguous_major = False
    if len(tied_major_list) > 1:
        ambiguous_major = True
        ref_base = None
        pos = None
        try:
            pos = int(get_position(row))
        except Exception:
            pos = None
        if feature_sequences is not None and pos is not None:
            ref_base = feature_sequences.get(pos)
        if ref_base in tied_major_list:
            major_allele = ref_base
        else:
            #Choose first in alphabetical order
            major_allele = sorted(tied_major_list)[0]
        major_count = base_counts[major_allele]
    else:
        # If no excact tie, use the major allele directly
        major_allele, major_count = sorted_bases[0]

    # Check for other ambiguous majors (~55%/45% major/minor frequencies)
    if total_count_no_gaps > 0:
        freqs_no_gaps = {b: c/total_count_no_gaps for b, c in nuc_counts.items()}
        # Sort by frequency to get top two
        sorted_by_freq = sorted(freqs_no_gaps.items(), key=lambda x: x[1], reverse=True)
        (b1, f1), (b2, f2) = sorted_by_freq[0], sorted_by_freq[1]
        if 0.45 <= f1 <= 0.55 and 0.45 <= f2 <= 0.55:
            ambiguous_major = True

    # Major allele frequencies
    major_freq_no_gaps = major_count / total_count_no_gaps if total_count_no_gaps > 0 else 0
    major_freq_no_gaps_str = "{:.1%}".format(major_freq_no_gaps) if total_count_no_gaps > 0 else "N/A"   
    
    # Identify minor alleles with any frequency
    minor_candidates = []
    for allele, count in nuc_counts.items():
        if allele == major_allele:
            continue
        if count > 0:
            freq_no_gaps = count / total_count_no_gaps
            minor_candidates.append((allele, freq_no_gaps))
    if minor_candidates:
        minor_allele, minor_freq_no_gaps = max(minor_candidates, key=lambda x: x[1])
        minor_freq_no_gaps_str = "{:.1%}".format(minor_freq_no_gaps)
    else:
        # No minor allele at all
        minor_allele = "N/A"
        minor_freq_no_gaps_str = "0.0%"

    return 'normal', [
        total_count_no_gaps,
        base_counts['gap'],
        major_allele,
        major_freq_no_gaps_str,
        ambiguous_major,
        minor_allele,
        minor_freq_no_gaps_str
    ]


def _save_main_output(all_data, output_file, intermediate=False, file_count=None):
    """Helper function to save the main DataFrame."""
    columns = [
        'sample',
        'pos',
        'read_depth_no_gaps',
        'gap_count',
        'maj_allele',
        'maj_allele_freq_no_gaps',
        'ambigous_major',
        'min_allele',
        'min_allele_freq_no_gaps'
    ]

    df = pd.DataFrame(all_data, columns=columns)

    if intermediate:
        path = output_file.replace('.csv', f'_intermediate_{file_count}.csv')
        df.to_csv(path, index=False, mode="w", chunksize=1000)
        print(f"Intermediate file saved to: {path}")
        return path
    else:
        df.to_csv(output_file, index=False, mode="w", chunksize=1000)
        print(f"Main output file saved to: {output_file}")
        return output_file
    

def merge_large_csvs(folder_path, final_csv, file_pattern):
    """Efficiently merge large intermediate CSVs without running out of memory and clean up after merging."""

    # Find all intermediate files matching the pattern
    all_files = sorted([f for f in os.listdir(folder_path) if file_pattern in f and f.endswith(".csv")])

    if not all_files:
        print(f"No intermediate files found to merge for {final_csv}.")
        return

    final_path = os.path.join(folder_path, final_csv)

    # Open final CSV in write mode
    with open(final_path, "w") as fout:
        header_written = False
        for filename in all_files:
            file_path = os.path.join(folder_path, filename)
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                for chunk in pd.read_csv(file_path, chunksize=1000):
                    chunk.to_csv(fout, index=False, header=not header_written, mode="a")
                    header_written = True

    print(f"Combined final output saved to: {final_path}")

    # Cleanup: Remove intermediate files
    print("Deleting intermediate files...")
    for filename in all_files:
        try:
            os.remove(os.path.join(folder_path, filename))
            print(f"Deleted: {filename}")
        except OSError as e:
            print(f"Could not delete {filename}: {e}")


# PROCESS BASEFREQS FILES #

def process_base_freq_files(input_folder, output_file, feature_sequences):
    """ Reads all *_BaseFreqs.csv in input_folder, calculates major/minor alleles, and writes output to output_file """
    all_data = []
    file_count = 0

    base_freq_files = [f for f in os.listdir(input_folder) if f.endswith("_BaseFreqs.csv")]

    for filename in base_freq_files:
        file_count += 1
        ID = re.split(r'[._]', filename)[0]
        path_to_file = os.path.join(input_folder, filename)
        print(f"Processing file {file_count}: {filename} in {input_folder}")
        try:
            base_freq_df = pd.read_csv(path_to_file)
            for _, row in base_freq_df.iterrows():
                row_type, data = calculate_alleles(row, feature_sequences)
                if row_type == 'no_coverage':
                    continue

                position = get_position(row)
                
                read_depth_no_gaps = data[0]
                # Keep only positions with sufficient depth
                if read_depth_no_gaps < MIN_READ_DEPTH:
                    continue 

                gap_count = data[1]
                maj_allele = data[2]
                maj_freq = data[3]
                ambiguous_major = data[4]
                min_allele = data[5]
                min_freq = data[6]

                all_data.append([
                    ID,
                    position,
                    read_depth_no_gaps,
                    gap_count,
                    maj_allele,
                    maj_freq,
                    ambiguous_major,
                    min_allele,
                    min_freq
                ])

            # intermediate save
            if file_count % intermediate_save_every == 0:
                print(f"Saving intermediate file after {file_count} files")
                _save_main_output(all_data, output_file, intermediate=True, file_count=file_count)

                # Clear the lists to free memory
                all_data =[]

        except Exception as e:
            print(f"Error processing file {filename}: {e}")

    # After processing all files, save the final leftover chunk
    print("Saving final chunk ...")
    _save_main_output(all_data, output_file, intermediate=True, file_count ='final')

    # Combine all intermediate output files to a single csv
    print("Combining all intermediate output files...")
    merge_large_csvs(os.path.dirname(output_file), "allele_freqs_output.csv", file_pattern="allele_freqs_output_intermediate_")


# MAIN FUNCTION #

def main():

    base_dir = '../../'

    fasta_file = os.path.join(base_dir, "ref", "MN908947.3.fasta")
    # Preprocess these once 
    feature_sequences = preprocess_fasta(fasta_file)

    # For each coverage threshold folder, process the sites in sites_to_run
    for threshold_folder in threshold_folders:
        print(f"\n{'='*25} THRESHOLD: {threshold_folder} {'='*25}\n")

        for site in sites_to_run:
            print(f"Processing site: {site} for threshold: {threshold_folder}")

            basefreq_folder = os.path.join(base_dir, "basefreqs_by_site_filtered", threshold_folder, site)
            output_folder = os.path.join(base_dir, "processed_data", "transmission_pair_examples", "variant_output", threshold_folder, site)
            os.makedirs(output_folder, exist_ok=True)

            output_file = os.path.join(output_folder, "allele_freqs_output.csv")

            # Process each site's BaseFreq files
            process_base_freq_files(basefreq_folder, output_file, feature_sequences)

    print("All done!")


if __name__ == "__main__":
    main()