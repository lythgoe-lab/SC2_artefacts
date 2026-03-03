"""
This script should ideally be implemented to the 1_process_basefreqs_major_minor_freqs_part1_cluster_script.py script, 
but was run after to convert massive CSV outputs to single Parquet files for easier handling and storage.

It reads in the allele frequency CSV outputs in chunks, processes each chunk to convert data types appropriately,
and writes them to a single Parquet file with Snappy compression.
"""


import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

def csv_to_single_parquet(csv_path, parquet_path, chunksize=100_000):
    writer = None

    for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunksize)):
        print(f"Processing chunk {i}...")

        # Convert % strings to numeric decimals
        for col in ["maj_allele_freq_no_gaps", "min_allele_freq_no_gaps"]:
            chunk[col] = (
                chunk[col]
                .astype(str)
                .str.rstrip("%")
            )
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce").fillna(0.0) / 100.0
            chunk[col] = chunk[col].astype("float32")

        # pos as integer and if not able to convert, drop the row and warn
        pos_numeric = pd.to_numeric(chunk["pos"], errors="coerce")
        bad_pos = pos_numeric.isna().sum()
        if bad_pos > 0:
            print(f"Warning: {bad_pos} rows with non-numeric pos in chunk {i}, dropping them.")
            chunk = chunk.loc[~pos_numeric.isna()].copy()
            pos_numeric = pos_numeric[~pos_numeric.isna()]
        chunk["pos"] = pos_numeric.astype("int32")

        # read_depth_no_gaps and gap_count
        for col in ["read_depth_no_gaps", "gap_count"]:
            num = pd.to_numeric(chunk[col], errors="coerce")
            bad = num.isna().sum()
            if bad > 0:
                print(f"Warning: {bad} rows with non-numeric {col} in chunk {i}, dropping them.")
                chunk = chunk.loc[~num.isna()].copy()
                num = num[~num.isna()]
            chunk[col] = num.astype("int32")

        if chunk["ambigous_major"].dtype != bool:
            chunk["ambigous_major"] = chunk["ambigous_major"].astype("bool")
            
        chunk["sample"] = chunk["sample"].astype("category")
        chunk["maj_allele"] = chunk["maj_allele"].astype("category")
        chunk["min_allele"] = chunk["min_allele"].astype("category")

        table = pa.Table.from_pandas(chunk, preserve_index=False)

        if writer is None:
            writer = pq.ParquetWriter(
                parquet_path,
                table.schema,
                compression="snappy"
            )

        writer.write_table(table)

    if writer is not None:
        writer.close()
        print(f"Written single Parquet file: {parquet_path}")
    else:
        print("No data written (empty CSV?).")

def main():
    base_dir = "processed_data/variant_output/10x"

    sites_to_convert = ["QEUH_ARTIC_Unknown_ILLUMINA","QEUH_Unknown_ILLUMINA"]

    for site in sites_to_convert:
        csv_path = os.path.join(base_dir, site, "allele_freqs_output.csv")
        parquet_path = os.path.join(base_dir, site, "allele_freqs_output.parquet")

        if not os.path.exists(csv_path):
            print(f"Skipping {site}: CSV not found at {csv_path}")
            continue

        print(f"\nConverting {csv_path} -> {parquet_path}")
        csv_to_single_parquet(csv_path, parquet_path, chunksize=100_000)

    print("\nAll done.")

if __name__ == "__main__":
    main()


    
