"""
This script processes all *_BaseFreqs.csv files within a specified input folder (filtered by coverage threshold and sequencing centre).
For each file, it uses the base counts to determine the major allele and its frequency and all minor alleles (above a frequency threshold) and their frequencies.
Each frequency is calculated both with and without gaps in the denominator (total read depth).
The output is a large CSV file compiling this summary information across all positions (one position per row) 
for each sample ID (derived from the BaseFreqs filename) within the selected input folder. 
In rare cases where there is an exact tie for the major allele, these are flagged and saved to a separate "tied_majors" CSV file with the relevant allele and frequency information.
"""

import pandas as pd
import os
import re
import glob
from Bio import SeqIO
from BCBio import GFF
from Bio.Seq import Seq
from datetime import datetime


# Select input folders (sequencing centres and protocols) to run
sites_to_run = ["BRBR_ARTIC_4.1_ILLUMINA", "LSPA_ARTIC_4.1_ILLUMINA", "MILK_ARTIC_3_ILLUMINA", 
                "NORT_ARTIC_3_ILLUMINA", "NORT_ARTIC_4.1_ILLUMINA", "NORT_ARTIC_4_ILLUMINA",
                "NORT_ARTIC_Unknown_ILLUMINA", "NORW_ARTIC_4.1_ILLUMINA", "OXON_VeSeq_ILLUMINA",
                "PHEC_ARTIC_3_ILLUMINA", "PHEC_ARTIC_Unknown_ILLUMINA", "QEUH_ARTIC_3_ILLUMINA",
                "QEUH_ARTIC_4.1_ILLUMINA", "QEUH_ARTIC_Unknown_ILLUMINA", "QEUH_Unknown_ILLUMINA"]

# Select coverage threshold folders to run (e.g. 1x, 10x, 100x or 1000x across 50% of the genome)
threshold_folders = ["10x"]


# Threshold for allele frequency
ALLELE_FREQUENCY_THRESHOLD = 0.02  # 2%
MAX_MINOR_ALLELES = 3  # Set the maximum number of minor alleles expected
intermediate_save_every = 500



# Pre-process and store GFF3 and FASTA data
def preprocess_gff_fasta(gff_file, fasta_file):
    """Read the GFF and FASTA once, extract relevant features to a dictionary."""
    fasta_sequences = SeqIO.to_dict(SeqIO.parse(fasta_file, "fasta"))
    reference_sequence = next(iter(fasta_sequences.values())).seq
    feature_sequences = {}

    with open(gff_file) as gff_handle:
        for rec in GFF.parse(gff_handle, base_dict=fasta_sequences):
            for feature in rec.features:
                if feature.type in ["gene", "CDS", "mature_protein_region_of_CDS"]:
                    start = int(feature.location.start)
                    end = int(feature.location.end)
                    feature_sequences[start, end] = str(reference_sequence[start:end])

    return feature_sequences


def get_position(row):
    """Return the value from the first column starting with 'position in' """
    for col in row.index:
        if col.lower().startswith("position in"):
            return row[col]
    raise ValueError("No column starting with 'position in' found in the row")


def get_gene_sequence(position, feature_sequences):
    """Return the gene sequence from feature_sequences if the position falls within a known feature"""
    for start, end in feature_sequences.keys():
        if start <= position <= end:
            return feature_sequences[start, end]
    return None


def translate_allele_to_amino_acid(position, allele, gene_sequence):
    """Translate a single-nucleotide change into the resulting amino acid (or skip if invalid)"""
    position = int(position)
    codon_start = (position - 1) % 3
    original_codon = gene_sequence[codon_start:codon_start + 3].replace('U', 'T')

    if '-' in original_codon or allele == '-':
        return 'Translation Skipped due to Gap'

    mutated_codon = original_codon[:codon_start] + allele.replace('U', 'T') + original_codon[codon_start + 1:]

    if len(mutated_codon) != 3 or not all(nuc in ['A', 'T', 'C', 'G'] for nuc in mutated_codon):
        return 'Invalid Codon'

    return str(Seq(mutated_codon).translate())


def calculate_alleles(row, feature_sequences):
    """Determine major and minor alleles (and their frequencies) for a given row of base counts."""
    base_counts = {
        'A': row['A count'],
        'C': row['C count'],
        'G': row['G count'],
        'T': row['T count'],
        'gap': row['gap count']
    }

    total_count = sum(base_counts.values())
    total_count_no_gaps = total_count - base_counts['gap']

    if total_count == 0:
        return 'no_coverage', []
    
    # Sort bases by count and find max frequency base
    sorted_bases = sorted(base_counts.items(), key=lambda x: x[1], reverse=True)
    max_freq = sorted_bases[0][1] / total_count if total_count > 0 else 0
   
    # Detect tied major alleles
    tied_major_list = [allele for allele, count in sorted_bases if count / total_count == max_freq]

    if len(tied_major_list) > 1:
        # if there is a tie for the major allele, return a special flag
        tied_major_data = [(allele, "{:.1%}".format(base_counts[allele] / total_count)) for allele in tied_major_list]
        return 'tied_major', tied_major_data

    # Determine the major allele (if no tie)
    major_allele, major_count = sorted_bases[0]
    major_freq_with_gaps = major_count / total_count if total_count > 0 else 0
    major_freq_without_gaps = major_count / total_count_no_gaps if total_count_no_gaps > 0 else 0

    # Convert to percentage
    major_freq_with_gaps = "{:.1%}".format(major_freq_with_gaps)
    major_freq_without_gaps = "{:.1%}".format(major_freq_without_gaps) if total_count_no_gaps > 0 else "N/A"
   
    
    # Identify minor alleles where frequency of above threshold
    minor_allele_frequencies = [
        (allele, count / total_count, count / total_count_no_gaps if total_count_no_gaps > 0 else 0)
        for allele, count in sorted_bases[1:] if count / total_count >= ALLELE_FREQUENCY_THRESHOLD
        ]

    # Prepare minor allele columns (up to 3 minor alleles)
    minor_alleles_list = []
    for allele, freq_with_gaps, freq_without_gaps in minor_allele_frequencies[:MAX_MINOR_ALLELES]:
        freq_without_gaps_str = "N/A" if allele == "gap" else "{:.1%}".format(freq_without_gaps)
        minor_alleles_list.extend([allele, "{:.1%}".format(freq_with_gaps), 
                                   freq_without_gaps_str])

    # Ensure the minoe allele list has 3 sets of values and otherwise fill with 'N/A'
    while len(minor_alleles_list) < 9:
        minor_alleles_list.extend(["N/A", "N/A", "N/A"])


    # Translate alleles to amino acids
    position = get_position(row)
    gene_sequence = get_gene_sequence(position, feature_sequences)
    major_amino_acid = translate_allele_to_amino_acid(position, major_allele, gene_sequence) if gene_sequence else "N/A"
    
    minor_amino_acids = [
        translate_allele_to_amino_acid(position, allele, gene_sequence) if allele not in ["N/A"] and gene_sequence else "N/A"
        for allele in minor_alleles_list[::3]]

    # Ensure exactly MAX_MINOR_ALLELES in minor_amino_acids
    while len(minor_amino_acids) < MAX_MINOR_ALLELES:
        minor_amino_acids.append("N/A")

    return 'normal', [
        total_count, total_count_no_gaps, major_allele, major_freq_with_gaps, major_freq_without_gaps, 
        *minor_alleles_list, major_amino_acid, *minor_amino_acids]



def _save_main_output(all_data, output_file, intermediate=False, file_count=None):
    """Helper function to save the main DataFrame."""
    columns = [
        'ID', 'Position',
        'A count', 'C count', 'G count', 'T count', 'gap count', 
        'read_depth', 'read_depth_no_gaps',
        'major_allele', 'percentage_with_gaps', 'percentage_without_gaps',
        'minor_allele1', 'minor_allele1_freq', 'minor_allele1_freq_no_gaps',
        'minor_allele2', 'minor_allele2_freq', 'minor_allele2_freq_no_gaps',
        'minor_allele3', 'minor_allele3_freq', 'minor_allele3_freq_no_gaps',
        'major_amino_acid', 'minor_amino_acid1', 'minor_amino_acid2', 'minor_amino_acid3'
    ]

    if len(columns) != 25:
        print(f"Warning: Expected 25 columns but found {len(columns)} in _save_main_output().")

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


def _save_tied_alleles(tied_data, output_file, intermediate=False, file_count=None):
    """Helper function to save the 'tied_majors' data if it exists."""
    if not tied_data:
        return
    
    # Clean each row: remove trailing empty strings
    cleaned_data = []
    for row in tied_data:
        # Remove trailing empty strings
        while row and row[-1] == '':
            row = row[:-1]
        cleaned_data.append(row)

    # Figure out max # of tied alleles across all rows
    # We always have [ID, Position], so subtract 2 from the length.
    max_tied_alleles = max(len(row) for row in cleaned_data) - 2

    # Build columns: ID, Position, plus Tied Allele columns
    columns = ['ID', 'Position'] + [
        f'Tied Allele {i + 1}' for i in range(max_tied_alleles)
    ]

    # Pad all rows so they match the max # of Tied Allele columns
    new_rows = []
    for row in tied_data:
        row_len = len(row) - 2
        # fill with 'N/A' if shorter
        missing = max_tied_alleles - row_len
        new_rows.append(row + ['N/A'] * missing)

    tied_df = pd.DataFrame(new_rows, columns=columns)


    # Pad all rows so they match the max # of Tied Allele columns
    new_rows = []
    for row in cleaned_data:
        row_len = len(row) - 2
        missing = max_tied_alleles - row_len
        new_rows.append(row + ['N/A'] * missing)


    if intermediate:
        path = output_file.replace('.csv', f'_tied_majors_intermediate_{file_count}.csv')
        tied_df.to_csv(path, index=False, mode="w", chunksize=1000)
        print(f"Intermediate tied majors file saved to: {path}")
        return path
    else:
        tied_outfile = output_file.replace('.csv', '_tied_majors.csv')
        tied_df.to_csv(tied_outfile, index=False, mode="w", chunksize=1000)
        print(f"Final tied output file saved to: {tied_outfile}")
        return tied_outfile
 

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



def process_base_freq_files(input_folder, output_file, feature_sequences):
    """ Reads all *_BaseFreqs.csv in input_folder, calculates major/minor alleles, writes output to output_file, and also generates a tied_majors file. """
    all_data = []
    tied_data = []
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
                position = get_position(row)
                if row_type == 'no_coverage':
                    continue
                elif row_type == 'tied_major':
                    # Handle tied major alleles by appending allele and count pairs
                    tied_data.append([ID, position] + [f"{allele}:{freq}" for allele, freq in data])
                else:
                # Otherwise save 'normal' major/minor allele data, if all minor alleles is not N/A
                    minor_alleles = data[5:14:3]  # Extract minor_allele1, minor_allele2, minor_allele3

                    if any(allele != "N/A" for allele in minor_alleles):
                        all_data.append([ID, position, row['A count'], row['C count'], row['G count'], row['T count'], row['gap count'], *data ])
                        
            # intermediate save
            if file_count % intermediate_save_every == 0:
                print(f"Saving intermediate file after {file_count} files")
                _save_main_output(all_data, output_file, intermediate=True, file_count=file_count)
                _save_tied_alleles(tied_data, output_file, intermediate = True, file_count=file_count)

                # Clear the lists to free memory
                all_data =[]
                tied_data =[]

        except Exception as e:
            print(f"Error processing file {filename}: {e}")

    # After processing all files, save the final leftover chunk
    print("Saving final chunk ...")
    _save_main_output(all_data, output_file, intermediate=True, file_count ='final')
    _save_tied_alleles(tied_data, output_file, intermediate =True, file_count='final')

    # Combine all intermediate output files to a single csv
    print("Combining all intermediate output files...")
    merge_large_csvs(os.path.dirname(output_file), "output_file.csv", file_pattern="output_file_intermediate_")
    merge_large_csvs(os.path.dirname(output_file), "output_file_tied_majors.csv", file_pattern="output_file_tied_majors_intermediate_")



def merge_site_files(file_paths, combined_csv):
    with open(combined_csv, "w") as fout:
        header_written = False
        for file_path in file_paths:
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                for chunk in pd.read_csv(file_path, chunksize=1000):
                    chunk.to_csv(fout, index=False, header=not header_written, mode="a")
                    header_written = True
    print(f"Combined file saved to: {combined_csv}")




def main():

    base_dir = '.'

    gff_file = os.path.join(base_dir, "ref", "sequence.gff3")
    fasta_file = os.path.join(base_dir, "ref", "MN908947.3.fasta")
    # Preprocess these once 
    feature_sequences = preprocess_gff_fasta(gff_file, fasta_file)

    # For each coverage threshold folder, process the sites in sites_to_run
    for threshold_folder in threshold_folders:
        print(f"\n{'='*25} THRESHOLD: {threshold_folder} {'='*25}\n")

        for site in sites_to_run:
            print(f"Processing site: {site} for threshold: {threshold_folder}")

            basefreq_folder = os.path.join(base_dir, "basefreqs_by_site_filtered", threshold_folder, site)
            output_folder = os.path.join(base_dir, "processed_data", threshold_folder, site)
            os.makedirs(output_folder, exist_ok=True)

            output_file = os.path.join(output_folder, "output_file.csv")

            # Process each site's BaseFreq files
            process_base_freq_files(basefreq_folder, output_file, feature_sequences)
        
        # Merge all site output files for each threshold
        print(f"\Merging outputs for all sites in threshold {threshold_folder} into combined files...\n")
        site_output_files = [os.path.join(base_dir, "processed_data", threshold_folder, site, "output_file.csv") for site in sites_to_run]
        site_tied_output_files = [os.path.join(base_dir, "processed_data", threshold_folder, site, "output_file_tied_majors.csv") for site in sites_to_run]
        combined_output = os.path.join(base_dir, "processed_data", threshold_folder, "all_ILLUMINA_output_file.csv")
        combined_tied_output = os.path.join(base_dir, "processed_data", threshold_folder, "all_ILLUMINA_output_file_tied_majors.csv")
        merge_site_files(site_output_files, combined_output)
        merge_site_files(site_tied_output_files, combined_tied_output)
        
    print("All done!")


if __name__ == "__main__":
    main()