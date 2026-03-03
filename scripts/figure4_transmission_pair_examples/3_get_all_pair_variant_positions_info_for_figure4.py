"""
This script extracts variant positions and frequencies for all samples in pairs of interest (typically high-confidence or random pairs).
It uses per-sample Parquet files as input and outputs a CSV file with extensive variant information for each pair, 
including categorising SNPs and iSNVs based on different MAF and depth thresholds, and flagging whether positions are masked.

Outputs contain classifications of SNPs and iSNV types (e.g. "snp_any" = any consensus change; "snp_isnv_site_maf" or "snp_isnv_1pct_maf" = SNP where the recipient has 
the source major allele as an iSNV at either the site-specific (from artefact analysis) or 1% MAF thresholds; or 
"shared_isnv_site_maf" or "shared_isnv_1pct_maf" = shared iSNV in source and recipient at site-specific or 1% MAF thresholds, etc).

"""

import os
import json
import ast
from functools import lru_cache
from typing import Callable
import pandas as pd
import numpy as np
from collections import defaultdict

base_dir = "../../"

#################################

# Specify category e.g. high_confidence or random  

CATEGORY = "high_confidence"

#################################

# Specify input and output paths for high-confidence or random pairs

PAIRS_CSV = os.path.join(
    base_dir,
    "processed_data", "transmission_pair_examples",
    "household_pairs_high_confidence_with_sequences_and_protocols.csv",
)

#PAIRS_CSV = os.path.join(
#    base_dir,
#    "processed_data", "transmission_pair_examples",
#    "random_pairs.csv",
#)


OUT_CSV = os.path.join(
    base_dir,
    "processed_data", "transmission_pair_examples",
    f"{CATEGORY}_pairs_all_variant_types_and_freqs.csv",
)

####################################

# Paths to MAF thresholds and masks per sequencing site, and to parquet base folder 

maf_df_path = os.path.join(
    base_dir,
    "processed_data",
    "10x",
    "final_maf_thresholds.csv",
)

mask_df_path = os.path.join(
    base_dir,
    "processed_data",
    "10x",
    "final_masks_all_centres.csv",
)

parquet_base = os.path.join(
    base_dir,
    "processed_data",
    "transmission_pair_examples",
    "variant_output",
    "10x",
)


######################################

# Constraints for variant extraction (coding region, depth thresholds, MAFs)

CODING_START = 265
CODING_END = 29674
DEPTH_THRESHOLDS = [100, 1000] # optionally include more than one depth threshold for analysis
DEFAULT_MAF = 0.01  # 1% default MAF as a "low" threshold or used when no site-specific threshold is available
MIN_MINOR_READS = 3  # minimum number of reads supporting minor allele to call iSNV

########################################

# function to normalise sequencing site names
def _norm_site(protocol_str) -> str:
    if not isinstance(protocol_str, str):
        return protocol_str
    s = protocol_str.strip()
    if s.endswith("_ILLUMINA"):
        s = s[: -len("_ILLUMINA")]
    for prefix in ("QEUH", "LSPA", "BRBR", "MILK"):
        if s.startswith(prefix):
            parts = s.split("_", 1)
            return "SANG_" + parts[1] if len(parts) > 1 else "SANG"
    return s

# function to parse mask positions from string representation
def parse_positions(x):
    if isinstance(x, (list, tuple, set)):
        return set(int(p) for p in x)
    if not isinstance(x, str):
        return set()
    s = x.strip()
    if not s:
        return set()
    try:
        return set(int(p) for p in json.loads(s))
    except Exception:
        try:
            return set(int(p) for p in ast.literal_eval(s))
        except Exception:
            return set()

# function to load sequencing site-specific MAF thresholds (from artefactual_sites_paper)
def load_site_maf_thresholds(path: str) -> dict:
    maf_df = pd.read_csv(path)
    maf_df["Site_norm"] = maf_df["Site"].map(_norm_site)
    maf_df["MAF_analysis"] = pd.to_numeric(maf_df["MAF_analysis"], errors="coerce")
    maf_df = maf_df.dropna(subset=["Site_norm", "MAF_analysis"]).copy()
    maf_df["maf_thr"] = maf_df["MAF_analysis"] / 100.0
    site_to_maf = dict(zip(maf_df["Site_norm"], maf_df["maf_thr"]))
    print(f"[INFO] Loaded {len(site_to_maf)} site-specific MAF thresholds")
    return site_to_maf

# function to load masks per sequencing site (from artefactual_sites_paper)
def load_mask_lookup():
    maf_df = pd.read_csv(maf_df_path)
    maf_df["Site_norm"] = maf_df["Site"].map(_norm_site)
    site_to_maskname = maf_df.set_index("Site_norm")["mask_name"].to_dict()

    masks = pd.read_csv(mask_df_path)
    masks = masks[~masks["mask"].astype(str).str.startswith("positions")].copy()
    masks["positions_set"] = masks["positions"].apply(parse_positions)
    masks["Centre_protocol_norm"] = masks["Centre_protocol"].map(_norm_site)
    mask_lookup = {
        (r.Centre_protocol_norm, r.mask): r.positions_set
        for r in masks.itertuples(index=False)
    }
    def mask_positions_for_site(site_norm: str) -> set[int]:
        name = site_to_maskname.get(site_norm)
        if name is None:
            return set()
        return mask_lookup.get((site_norm, name), set())

    return mask_positions_for_site



# Function to deduplicate positions in per-sample parquet slices (as some positions were accidentally duplicated for some samples)
def _dedup_positions(df: pd.DataFrame, protocol: str, sample: str) -> pd.DataFrame:
    if df.empty:
        return df
    if df["pos"].is_unique:
        return df
    dup_pos = df.loc[df["pos"].duplicated(keep=False), "pos"].unique()
    for p in dup_pos:
        sub = df[df["pos"] == p]
        if sub.drop_duplicates().shape[0] != 1:
            print("\n[ERROR] Non-identical duplicate positions found in per-sample parquet slice.")
            print(f"protocol={protocol}")
            print(f"sample ={sample}")
            print(f"pos ={int(p)}")
            with pd.option_context("display.max_columns", 200, "display.width", 220):
                print(sub.to_string(index=False))
            raise ValueError(
                f"[Error] Run stopped: Non-identical duplicate rows for sample={sample} protocol={protocol} pos={int(p)}"
            )
    return df.drop_duplicates(subset=["pos"], keep="first").copy()

# function to back-calculate minor and major read counts to per-sample dataframe from MAF and depth
def _add_read_counts(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        df["minor_reads"] = pd.Series(dtype="int32")
        df["major_reads"] = pd.Series(dtype="int32")
        return df

    depth = pd.to_numeric(df["read_depth_no_gaps"], errors="coerce").fillna(0).astype("int32")
    minor_freq = pd.to_numeric(df["min_allele_freq_no_gaps"], errors="coerce").fillna(0.0)
    maj_freq = pd.to_numeric(df["maj_allele_freq_no_gaps"], errors="coerce").fillna(0.0)

    minor_reads = np.floor(minor_freq.to_numpy() * depth.to_numpy()).astype("int32")
    major_reads = np.floor(maj_freq.to_numpy() * depth.to_numpy()).astype("int32")

    df = df.copy()
    df["read_depth_no_gaps"] = depth
    df["minor_reads"] = minor_reads
    df["major_reads"] = major_reads
    return df

# function to load per-sample variant data from parquet files (with caching for speed)
@lru_cache(maxsize=200)
def load_sample(protocol: str, sample: str) -> pd.DataFrame | None:
    folder = protocol.strip()
    parquet_path = os.path.join(parquet_base, folder, "allele_freqs_output.parquet")

    if not os.path.exists(parquet_path):
        print(f"[WARNING] Missing parquet for protocol {protocol}: {parquet_path}")
        return None

    base_cols = [
        "sample",
        "pos",
        "read_depth_no_gaps",
        "maj_allele",
        "min_allele",
        "maj_allele_freq_no_gaps",
        "min_allele_freq_no_gaps",
        "ambigous_major",
    ]

    df = pd.read_parquet(
        parquet_path,
        columns=base_cols,
        filters=[("sample", "=", sample)],
    )
    if df.empty:
        return None

    df = df.copy()
    df["pos"] = pd.to_numeric(df["pos"], errors="coerce")
    df = df.dropna(subset=["pos"])
    df["pos"] = df["pos"].astype("int32")

    df = df[(df["pos"] >= CODING_START) & (df["pos"] <= CODING_END)]
    if df.empty:
        return None

    df["maj_allele_freq_no_gaps"] = pd.to_numeric(
        df["maj_allele_freq_no_gaps"], errors="coerce"
    ).fillna(0.0)
    df["min_allele_freq_no_gaps"] = pd.to_numeric(
        df["min_allele_freq_no_gaps"], errors="coerce"
    ).fillna(0.0)
    df["read_depth_no_gaps"] = pd.to_numeric(
        df["read_depth_no_gaps"], errors="coerce"
    ).fillna(0).astype("int32")

    df = _add_read_counts(df)

    df["ambigous_major"] = df["ambigous_major"].fillna(False).astype(bool)
    df = _dedup_positions(df, protocol=protocol, sample=sample)
    return df

# function to load transmission pairs from CSV
def load_pairs(pairs_csv: str, category: str) -> pd.DataFrame:
    pairs_df = pd.read_csv(pairs_csv, dtype=str).fillna("")
    # Deterministic pair order so pair_n=i is stable across runs
    sort_cols = []
    # High-confidence-style household column (just one)
    if "hh" in pairs_df.columns:
        sort_cols.append("hh")
    # Random-style household columns (two households)
    if "hh1" in pairs_df.columns:
        sort_cols.append("hh1")
    if "hh2" in pairs_df.columns:
        sort_cols.append("hh2")
    # Core identifiers
    for c in ["sample1", "sample2", "protocol1", "protocol2"]:
        if c in pairs_df.columns:
            sort_cols.append(c)
    # dates if present
    for c in ["collection_date1", "collection_date2"]:
        if c in pairs_df.columns:
            sort_cols.append(c)
    if sort_cols:
        pairs_df = pairs_df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    print(f"[INFO] Loaded {len(pairs_df)} pairs from {pairs_csv} (category={category})")
    return pairs_df



# function to build the full pair variant table from per-sample parquet data
def build_pair_variant_table(
    pairs_df: pd.DataFrame,
    depth_thresholds: list[int],
    site_to_maf: dict,
    mask_positions_for_site: Callable[[str], set[int]],
    default_maf: float,
    category: str,
) -> pd.DataFrame:
    records = []
    has_hh = "hh" in pairs_df.columns
    has_hh1 = "hh1" in pairs_df.columns
    has_hh2 = "hh2" in pairs_df.columns


    for i, row in enumerate(pairs_df.itertuples(index=False), start=1):
        pair_n = i # assign pair number to each pair
        hh = getattr(row, "hh", "") if has_hh else ""
        hh1 = getattr(row, "hh1", "") if has_hh1 else ""
        hh2 = getattr(row, "hh2", "") if has_hh2 else ""

        sample1 = row.sample1
        sample2 = row.sample2
        protocol1 = row.protocol1
        protocol2 = row.protocol2
        date1 = getattr(row, "collection_date1", "")
        date2 = getattr(row, "collection_date2", "")

        df1 = load_sample(protocol1, sample1)
        df2 = load_sample(protocol2, sample2)

        if df1 is None or df2 is None or df1.empty or df2.empty:
            continue

        site_norm1 = _norm_site(protocol1) # "site" refers to sequencing site/protocol
        site_norm2 = _norm_site(protocol2)
        maf_src = site_to_maf.get(site_norm1, default_maf) # get site-specific MAF threshold for source or use default
        maf_rec = site_to_maf.get(site_norm2, default_maf) # get site-specific MAF threshold for recipient or use default
        maskset1 = mask_positions_for_site(site_norm1) # get mask positions for source sequencing site
        maskset2 = mask_positions_for_site(site_norm2) # get mask positions for recipient sequencing site

        # process each depth threshold separately
        for depth_thr in depth_thresholds:
            d1 = df1[df1["read_depth_no_gaps"] >= depth_thr].copy()  # use depth without counting gaps
            d2 = df2[df2["read_depth_no_gaps"] >= depth_thr].copy()
            if d1.empty or d2.empty:
                continue

            merged = pd.merge(d1, d2, on="pos", suffixes=("_src", "_rec"))
            if merged.empty:
                continue

            # remove positions where major allele is a gap in either sample
            merged = merged[
                (merged["maj_allele_src"] != "gap")
                & (merged["maj_allele_rec"] != "gap")
            ].copy()
            if merged.empty:
                continue

            merged["ambiguous_maj_src"] = merged["ambigous_major_src"].astype(bool)
            merged["ambiguous_maj_rec"] = merged["ambigous_major_rec"].astype(bool)

            merged = merged.rename(
                columns={
                    "maj_allele_src": "maj_src",
                    "min_allele_src": "min_src",
                    "maj_allele_rec": "maj_rec",
                    "min_allele_rec": "min_rec",
                    "read_depth_no_gaps_src": "read_depth_src",
                    "read_depth_no_gaps_rec": "read_depth_rec",
                }
            )

            merged["minor_freq_src"] = merged["min_allele_freq_no_gaps_src"]
            merged["minor_freq_rec"] = merged["min_allele_freq_no_gaps_rec"]
            merged["maj_freq_src"] = merged["maj_allele_freq_no_gaps_src"]
            merged["maj_freq_rec"] = merged["maj_allele_freq_no_gaps_rec"]

            merged["consensus_change"] = merged["maj_src"] != merged["maj_rec"]  # here, consensus change means major allele change between source and recipient, even if frequencies are close to 50%

            # to call an iSNV, require MAF threshold AND >= MIN_MINOR_READS
            isnv_source_site_maf = (merged["minor_freq_src"] >= maf_src) & (merged["minor_reads_src"] >= MIN_MINOR_READS)
            isnv_recipient_site_maf = (merged["minor_freq_rec"] >= maf_rec) & (merged["minor_reads_rec"] >= MIN_MINOR_READS)

            isnv_source_1pct_maf = (merged["minor_freq_src"] >= DEFAULT_MAF) & (merged["minor_reads_src"] >= MIN_MINOR_READS)
            isnv_recipient_1pct_maf = (merged["minor_freq_rec"] >= DEFAULT_MAF) & (merged["minor_reads_rec"] >= MIN_MINOR_READS)

            merged["isnv_source_site_maf"] = isnv_source_site_maf
            merged["isnv_recipient_site_maf"] = isnv_recipient_site_maf
            merged["isnv_source_1pct_maf"] = isnv_source_1pct_maf
            merged["isnv_recipient_1pct_maf"] = isnv_recipient_1pct_maf

            # keep only positions with some signal - either consensus change or iSNV in source and/or recipient
            has_signal = merged["consensus_change"] | isnv_source_1pct_maf | isnv_recipient_1pct_maf
            merged = merged[has_signal].copy()
            if merged.empty:
                continue

            merged["masked_src"] = merged["pos"].isin(maskset1)
            merged["masked_rec"] = merged["pos"].isin(maskset2)

            # define SNP and iSNV categories to add as summary columns

            # "any" SNP - regardless of iSNV status
            merged["snp_any"] = merged["consensus_change"]
            # SNP with no diversity in source or recipient at a 1% MAF threshold
            merged["snp_no_isnv_1pct_maf"] = (
                merged["consensus_change"]
                & (merged["minor_freq_src"] < DEFAULT_MAF)
                & (merged["minor_freq_rec"] < DEFAULT_MAF)
            )
            # SNP with no diversity in source or recipient at site-specific MAF thresholds
            merged["snp_no_isnv_site_maf"] = (
                merged["consensus_change"]
                & (merged["minor_freq_src"] < maf_src)
                & (merged["minor_freq_rec"] < maf_rec)
            )
            # iSNV in both source and recipient at 1% MAF threshold, with same minor allele
            shared_isnv_1pct_maf = (
                isnv_source_1pct_maf
                & isnv_recipient_1pct_maf
                & (merged["min_src"] == merged["min_rec"])
            )
            # iSNV in both source and recipient at site-specific MAF thresholds, with same minor allele
            shared_isnv_site_maf = (
                isnv_source_site_maf
                & isnv_recipient_site_maf
                & (merged["min_src"] == merged["min_rec"])
            )
            merged["shared_isnv_1pct_maf"] = shared_isnv_1pct_maf
            merged["shared_isnv_site_maf"] = shared_isnv_site_maf

            # SNP where variant is iSNV in source at 1% MAF threshold
            merged["isnv_snp_1pct_maf"] = (
                merged["consensus_change"]
                & isnv_source_1pct_maf
                & (merged["min_src"] == merged["maj_rec"])
            )
            # SNP where variant is iSNV in recipient at 1% MAF threshold
            merged["snp_isnv_1pct_maf"] = (
                merged["consensus_change"]
                & isnv_recipient_1pct_maf
                & (merged["maj_src"] == merged["min_rec"])
            )
            # SNP where variant is iSNV in source at site-specific MAF threshold
            merged["isnv_snp_site_maf"] = (
                merged["consensus_change"]
                & isnv_source_site_maf
                & (merged["min_src"] == merged["maj_rec"])
            )
            # SNP where variant is iSNV in recipient at site-specific MAF threshold
            merged["snp_isnv_site_maf"] = (
                merged["consensus_change"]
                & isnv_recipient_site_maf
                & (merged["maj_src"] == merged["min_rec"])
            )
            # iSNV only in source or only in recipient at 1% MAF threshold
            merged["isnv_src_only_1pct_maf"] = isnv_source_1pct_maf & ~shared_isnv_1pct_maf
            merged["isnv_rec_only_1pct_maf"] = isnv_recipient_1pct_maf & ~shared_isnv_1pct_maf

            # iSNV only in source or only in recipient at site-specific MAF thresholds
            merged["isnv_src_only_site_maf"] = isnv_source_site_maf & ~shared_isnv_site_maf
            merged["isnv_rec_only_site_maf"] = isnv_recipient_site_maf & ~shared_isnv_site_maf

            # metadata
            merged["pair_n"] = pair_n
            if has_hh:
                merged["hh"] = hh
            if has_hh1:
                merged["hh1"] = hh1
            if has_hh2:
                merged["hh2"] = hh2

            merged["ID_src"] = sample1
            merged["ID_rec"] = sample2
            merged["date_src"] = date1
            merged["date_rec"] = date2
            merged["site_src"] = protocol1
            merged["site_rec"] = protocol2
            merged["depth_thresh"] = f"{depth_thr}x"
            merged["category"] = category

            # final columns to keep in output
            keep_cols = [
                "pair_n",
                "pos",
                "ID_src",
                "ID_rec",
                "date_src",
                "date_rec",
                "site_src",
                "site_rec",
                "read_depth_src",
                "read_depth_rec",
                "maj_src",
                "maj_rec",
                "maj_freq_src",
                "maj_freq_rec",
                "min_src",
                "min_rec",
                "minor_freq_src",
                "minor_freq_rec",
                "ambiguous_maj_src",
                "ambiguous_maj_rec",
                "consensus_change",
                "snp_any",
                "snp_no_isnv_1pct_maf",
                "snp_no_isnv_site_maf",
                "isnv_source_1pct_maf",
                "isnv_recipient_1pct_maf",
                "isnv_source_site_maf",
                "isnv_recipient_site_maf",
                "isnv_snp_1pct_maf",
                "snp_isnv_1pct_maf",
                "shared_isnv_1pct_maf",
                "isnv_snp_site_maf",
                "snp_isnv_site_maf",
                "shared_isnv_site_maf",
                "isnv_src_only_1pct_maf",
                "isnv_rec_only_1pct_maf",
                "isnv_src_only_site_maf",
                "isnv_rec_only_site_maf",
                "masked_src",
                "masked_rec",
                "depth_thresh",
                "category",
            ]
            prefix_cols = []
            if has_hh:
                prefix_cols.append("hh")
            if has_hh1:
                prefix_cols.append("hh1")
            if has_hh2:
                prefix_cols.append("hh2")

            keep_cols = prefix_cols + keep_cols

            records.append(merged[keep_cols])

    if not records:
        print("[WARNING] No variant rows constructed from parquet")
        return pd.DataFrame()

    df_out = pd.concat(records, ignore_index=True)

    # Sort
    sort_cols = []
    if "hh" in df_out.columns:
        sort_cols.append("hh")
    if "hh1" in df_out.columns:
        sort_cols.append("hh1")
    if "hh2" in df_out.columns:
        sort_cols.append("hh2")
    sort_cols += ["pair_n", "pos"]

    df_out = df_out.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)


    print(
        f"[INFO] Built variant table with {len(df_out)} rows across {df_out['pair_n'].nunique()} pairs"
        + (f" in {df_out['hh'].nunique()} households" if "hh" in df_out.columns else "")
    )
    return df_out



def main():
    site_to_maf = load_site_maf_thresholds(maf_df_path)
    mask_positions_for_site = load_mask_lookup()

    pairs_df = load_pairs(PAIRS_CSV, category=CATEGORY)

    df = build_pair_variant_table(
        pairs_df=pairs_df,
        depth_thresholds=DEPTH_THRESHOLDS,
        site_to_maf=site_to_maf,
        mask_positions_for_site=mask_positions_for_site,
        default_maf=DEFAULT_MAF,
        category=CATEGORY,
    )


    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"[INFO] Saved {len(df)} rows to {OUT_CSV}")

    del df, mask_positions_for_site, site_to_maf, pairs_df



if __name__ == "__main__":
    main()
