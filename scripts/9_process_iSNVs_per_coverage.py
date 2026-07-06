""" This script builds a table of iSNVs per sample and samples with coverage >= 1000x for each site and various masking/MAF conditions.
The iSNV information is taken from the total iSNVs per sample and MAF parquet files, created using the same set of samples as the rest of the artefact analysis,
but samples are only included if they have at least one position with an iSNV called above 2% MAF.
The depth information per position is taken from allele_freqs_output.parquet files, which were created for another analysis,
and since there are slight differences in the sample sets used for these analyses (max ~200 samples), the script uses a sample overlap file to identify which samples 
are only present in the artefact analysis, only present in the pairs analysis, or present in both to infer the right amount of samples with enough depth but no iSNVs (in contrast to
samples that were never present in the artefact analysis)

The script processes each site in batches to avoid memory issues, and outputs a table with one row per sample and condition, with the number of iSNVs for that sample and condition, 
the total number of positions with >= 1000x depth for that sample, and the reason for zero iSNVs if there are zero (no positions with >= 1000x depth, present in both inputs but missing 
from iSNV summary, or explicitly zero in iSNV summary)
"""

import os
import glob
import gc
import pandas as pd
import numpy as np
import pyarrow.dataset as ds
from collections import defaultdict


base_dir = "../../"

total_isnvs_parquet_base = os.path.join(
    base_dir,
    "artefactual_sites_paper",
    "processed_data",
    "10x",
    "total_isnvs_per_sample_and_maf_all_masks_depth_1000x"
)

all_depths_parquet_base = os.path.join(
    base_dir,
    "transmission_pairs",
    "processed_data",
    "variant_output",
    "10x",
)

final_maf_thresholds_file = os.path.join(
    base_dir,
    "artefactual_sites_paper",
    "processed_data",
    "10x",
    "final_maf_thresholds.csv"
)

sample_overlap_file = os.path.join(
    base_dir,
    "artefactual_sites_paper",
    "artefact_pairs_sample_mismatches.csv"
)

output_file = os.path.join(
    base_dir,
    "artefactual_sites_paper",
    "processed_data",
    "10x",
    "sample_iSNVs_per_depth.csv")

MIN_DEPTH = 1000
BATCH_SIZE = 250_000

DEFAULT_UNMASKED_MAF = "3"

DEPTH_SAMPLE_COL = "sample"
DEPTH_COL = "read_depth_no_gaps"

ISNV_SAMPLE_COL = "ID"
SITE_COL = "Site"
MASK_COL = "Mask_Condition"
MAF_COL = "MAF_Threshold_Percent"
ISNV_COL = "iSNV_Count"


SITE_GROUPS = [
    {
        "row_label": "SANG ARTIC 4.1",
        "sample_prefixes": ["BRBR", "LSPA", "QEUH"],
        "site_names": [
            "BRBR_ARTIC_4.1_Illumina",
            "LSPA_ARTIC_4.1_Illumina",
            "QEUH_ARTIC_4.1_Illumina",
        ],
        "threshold_site_name": "SANG_ARTIC_4.1",
    },
    {
        "row_label": "SANG ARTIC 3",
        "sample_prefixes": ["MILK", "QEUH"],
        "site_names": [
            "MILK_ARTIC_3_Illumina",
            "QEUH_ARTIC_3_Illumina",
        ],
        "threshold_site_name": "SANG_ARTIC_3",
    },
    {
        "row_label": "NORT ARTIC 3",
        "sample_prefixes": ["NORT"],
        "site_names": ["NORT_ARTIC_3_Illumina"],
        "threshold_site_name": "NORT_ARTIC_3",
    },
    {
        "row_label": "NORT ARTIC 4",
        "sample_prefixes": ["NORT"],
        "site_names": ["NORT_ARTIC_4_Illumina"],
        "threshold_site_name": "NORT_ARTIC_4",
    },
    {
        "row_label": "NORT ARTIC 4.1",
        "sample_prefixes": ["NORT"],
        "site_names": ["NORT_ARTIC_4.1_Illumina"],
        "threshold_site_name": "NORT_ARTIC_4.1",
    },
    {
        "row_label": "NORW ARTIC ?",
        "sample_prefixes": ["NORW"],
        "site_names": ["NORW_ARTIC_Unknown_Illumina"],
        "threshold_site_name": "NORW_ARTIC_4.1",
        "overlap_folder_names": [
            "NORW_ARTIC_Unknown_ILLUMINA",
            "NORW_ARTIC_4.1_ILLUMINA",
        ],
        "ignore_no_overlap": True,
    },
    {
        "row_label": "OXON ve-SEQ",
        "sample_prefixes": ["OXON"],
        "site_names": ["OXON_VeSeq_Illumina"],
        "threshold_site_name": "OXON_VeSeq",
    },
    {
        "row_label": "PHEC ARTIC 3",
        "sample_prefixes": ["PHEC"],
        "site_names": ["PHEC_ARTIC_3_Illumina"],
        "threshold_site_name": "PHEC_ARTIC_3",
    },
]


def clean_sample_name(x):
    if pd.isna(x):
        return None
    x = str(x).strip()
    if x == "":
        return None
    return x.replace("*", "").strip()


def read_sample_overlap_file(overlap_file):
    overlap = pd.read_csv(overlap_file)
    overlap["folder_name"] = overlap["folder_name"].astype(str).str.strip().str.upper()

    overlap["sample_only_in_artefacts_clean"] = (
        overlap["sample_only_in_artefacts"]
        .apply(clean_sample_name)
    )
    overlap["sample_only_in_pairs_clean"] = (
        overlap["sample_only_in_pairs"]
        .apply(clean_sample_name)
    )
    overlap["different_version"] = (
        overlap["sample_only_in_artefacts"].astype(str).str.contains(r"\*", regex=True)
        | overlap["sample_only_in_pairs"].astype(str).str.contains(r"\*", regex=True)
    )
    return overlap


def get_overlap_sets_for_site_group(site_group, overlap_df):
    folder_names_for_overlap = site_group.get(
        "overlap_folder_names",
        site_group["site_names"]
    )
    folder_keys = [
        str(s).strip().upper()
        for s in folder_names_for_overlap
    ]
    site_overlap = overlap_df[overlap_df["folder_name"].isin(folder_keys)].copy()

    pairs_only = set(
        site_overlap["sample_only_in_pairs_clean"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    artefacts_only = set(
        site_overlap["sample_only_in_artefacts_clean"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    different_version_pairs = set(
        site_overlap.loc[
            site_overlap["different_version"],
            "sample_only_in_pairs_clean"
        ]
        .dropna()
        .astype(str)
        .str.strip()
    )

    different_version_artefacts = set(
        site_overlap.loc[
            site_overlap["different_version"],
            "sample_only_in_artefacts_clean"
        ]
        .dropna()
        .astype(str)
        .str.strip()
    )

    return {
        "pairs_only_true": pairs_only - different_version_pairs,
        "artefacts_only_true": artefacts_only - different_version_artefacts,
        "different_version_pairs": different_version_pairs,
        "different_version_artefacts": different_version_artefacts,
    }


def read_final_thresholds_all(thresholds_file):
    thresholds = pd.read_csv(thresholds_file)

    thresholds["Site"] = thresholds["Site"].astype(str).str.strip()
    thresholds["mask_name"] = thresholds["mask_name"].astype(str).str.strip()
    thresholds["MAF_mask"] = thresholds["MAF_mask"].astype(str).str.strip()
    thresholds["MAF_analysis"] = thresholds["MAF_analysis"].astype(str).str.strip()

    return thresholds


def get_threshold_info(thresholds, threshold_site_name):
    row = thresholds[thresholds["Site"] == threshold_site_name].copy()

    if row.empty:
        raise ValueError(f"No threshold row found for Site == {threshold_site_name}")

    if len(row) > 1:
        raise ValueError(f"Multiple threshold rows found for Site == {threshold_site_name}")

    return row.iloc[0].to_dict()


def get_analysis_conditions(threshold_info):
    mask_name = str(threshold_info["mask_name"]).strip()
    maf_mask = str(threshold_info["MAF_mask"]).strip()
    maf_analysis = str(threshold_info["MAF_analysis"]).strip()

    return pd.DataFrame([
        {
            "Masked": False,
            MASK_COL: "no_mask",
            MAF_COL: DEFAULT_UNMASKED_MAF,
        },
        {
            "Masked": False,
            MASK_COL: "no_mask",
            MAF_COL: maf_mask,
        },
        {
            "Masked": False,
            MASK_COL: "no_mask",
            MAF_COL: maf_analysis,
        },
        {
            "Masked": True,
            MASK_COL: mask_name,
            MAF_COL: maf_mask,
        },
        {
            "Masked": True,
            MASK_COL: mask_name,
            MAF_COL: maf_analysis,
        },
    ]).drop_duplicates(ignore_index=True)


def iter_parquet_batches(path, columns):
    dataset = ds.dataset(path, format="parquet")

    for batch in dataset.to_batches(columns=columns, batch_size=BATCH_SIZE):
        yield batch.to_pandas()


def count_depth_samples_and_callable_positions(depth_base, site_names, sample_prefixes, min_depth):
    all_samples = set()
    callable_counts = {}

    last_pos_seen = {}

    for site_name in site_names:
        depth_path = os.path.join(
            depth_base,
            site_name,
            "allele_freqs_output.parquet"
        )

        print(f"Processing depth: {site_name}")

        for chunk in iter_parquet_batches(
            depth_path,
            columns=[DEPTH_SAMPLE_COL, "pos", DEPTH_COL]
        ):
            chunk[DEPTH_SAMPLE_COL] = chunk[DEPTH_SAMPLE_COL].astype(str).str.strip()

            prefix_mask = chunk[DEPTH_SAMPLE_COL].str.startswith(tuple(sample_prefixes))
            chunk = chunk[prefix_mask]

            if chunk.empty:
                continue

            all_samples.update(chunk[DEPTH_SAMPLE_COL].dropna().unique())

            chunk = chunk[chunk[DEPTH_COL] >= min_depth]

            if chunk.empty:
                continue

            for sample, pos in zip(chunk[DEPTH_SAMPLE_COL], chunk["pos"]):
                previous_pos = last_pos_seen.get(sample)

                if previous_pos is None or pos > previous_pos:
                    callable_counts[sample] = callable_counts.get(sample, 0) + 1
                    last_pos_seen[sample] = pos
                else:
                    # Position reset or duplicate block; ignore
                    continue

            del chunk
            gc.collect()

    full_depth_df = pd.DataFrame({
        DEPTH_SAMPLE_COL: sorted(all_samples)
    })

    count_df = pd.DataFrame({
        DEPTH_SAMPLE_COL: list(callable_counts.keys()),
        "total_positions_1000x": list(callable_counts.values()),
    })

    full_depth_df = full_depth_df.merge(
        count_df,
        on=DEPTH_SAMPLE_COL,
        how="left"
    )

    full_depth_df["total_positions_1000x"] = (
        full_depth_df["total_positions_1000x"]
        .fillna(0)
        .astype(int)
    )

    full_depth_df["has_1000x_position"] = (
        full_depth_df["total_positions_1000x"] > 0
    )

    return full_depth_df


def load_isnv_condition_counts(isnv_base, sample_set, site, condition_df):
    if len(sample_set) == 0:
        return pd.DataFrame(columns=[ISNV_SAMPLE_COL, "Masked", MAF_COL, ISNV_COL])

    parquet_files = sorted(glob.glob(os.path.join(isnv_base, "*.parquet")))

    if len(parquet_files) == 0:
        raise FileNotFoundError(f"No iSNV parquet files found in {isnv_base}")

    condition_lookup = condition_df.copy()
    condition_lookup[SITE_COL] = site

    parts = []

    for parquet_file in parquet_files:
        dataset = ds.dataset(parquet_file, format="parquet")

        for batch in dataset.to_batches(
            columns=[ISNV_SAMPLE_COL, SITE_COL, MASK_COL, MAF_COL, ISNV_COL],
            batch_size=BATCH_SIZE
        ):
            chunk = batch.to_pandas()

            chunk[ISNV_SAMPLE_COL] = chunk[ISNV_SAMPLE_COL].astype(str).str.strip()
            chunk[SITE_COL] = chunk[SITE_COL].astype(str).str.strip()
            chunk[MASK_COL] = chunk[MASK_COL].astype(str).str.strip()
            chunk[MAF_COL] = chunk[MAF_COL].astype(str).str.strip()

            chunk = chunk[
                (chunk[ISNV_SAMPLE_COL].isin(sample_set))
                & (chunk[SITE_COL] == site)
            ]

            if chunk.empty:
                continue

            chunk = chunk.merge(
                condition_lookup,
                on=[SITE_COL, MASK_COL, MAF_COL],
                how="inner"
            )

            if chunk.empty:
                continue

            parts.append(
                chunk[[ISNV_SAMPLE_COL, "Masked", MAF_COL, ISNV_COL]]
            )

            del chunk
            gc.collect()

    if len(parts) == 0:
        return pd.DataFrame(columns=[ISNV_SAMPLE_COL, "Masked", MAF_COL, ISNV_COL])

    out = pd.concat(parts, ignore_index=True)

    out = (
        out
        .groupby([ISNV_SAMPLE_COL, "Masked", MAF_COL], as_index=False)[ISNV_COL]
        .max()
    )

    return out


def build_sample_condition_table(site_group, thresholds, overlap_df):
    threshold_info = get_threshold_info(
        thresholds=thresholds,
        threshold_site_name=site_group["threshold_site_name"]
    )

    site = str(threshold_info["Site"]).strip()
    condition_df = get_analysis_conditions(threshold_info)

    full_depth_df = count_depth_samples_and_callable_positions(
        depth_base=all_depths_parquet_base,
        site_names=site_group["site_names"],
        sample_prefixes=site_group["sample_prefixes"],
        min_depth=MIN_DEPTH
    )

    overlap_sets = get_overlap_sets_for_site_group(site_group, overlap_df)

    if site_group.get("ignore_no_overlap", False):
        sample_df = full_depth_df.copy()
    else:
        sample_df = full_depth_df[
            ~full_depth_df[DEPTH_SAMPLE_COL].isin(overlap_sets["pairs_only_true"])
        ].copy()

    callable_sample_set = set(
        sample_df.loc[
            sample_df["has_1000x_position"],
            DEPTH_SAMPLE_COL
        ]
    )

    isnv_counts = load_isnv_condition_counts(
        isnv_base=total_isnvs_parquet_base,
        sample_set=callable_sample_set,
        site=site,
        condition_df=condition_df
    )

    sample_conditions = sample_df.merge(
        condition_df,
        how="cross"
    )

    sample_conditions[SITE_COL] = site
    sample_conditions["row_label"] = site_group["row_label"]

    sample_conditions = sample_conditions.merge(
        isnv_counts,
        left_on=[DEPTH_SAMPLE_COL, "Masked", MAF_COL],
        right_on=[ISNV_SAMPLE_COL, "Masked", MAF_COL],
        how="left"
    )

    sample_conditions["iSNV_missing_from_summary"] = sample_conditions[ISNV_COL].isna()
    sample_conditions[ISNV_COL] = sample_conditions[ISNV_COL].fillna(0).astype(int)

    sample_conditions["note"] = np.select(
        [
            sample_conditions["total_positions_1000x"].eq(0),
            sample_conditions["iSNV_missing_from_summary"],
            sample_conditions[ISNV_COL].eq(0),
        ],
        [
            "no_1000x_positions",
            "in_both_inputs_but_absent_from_isnv_summary",
            "zero_in_isnv_summary",
        ],
        default=pd.NA
    )

    sample_conditions = sample_conditions.rename(
        columns={
            MAF_COL: "MAF",
        }
    )

    keep_cols = [
        "row_label",
        DEPTH_SAMPLE_COL,
        "total_positions_1000x",
        SITE_COL,
        "Masked",
        "MAF",
        ISNV_COL,
        "note",
    ]

    return sample_conditions[keep_cols]


def main():
    thresholds = read_final_thresholds_all(final_maf_thresholds_file)
    overlap_df = read_sample_overlap_file(sample_overlap_file)

    if os.path.exists(output_file):
        os.remove(output_file)

    header_written = False

    for site_group in SITE_GROUPS:
        print(f"Processing {site_group['row_label']}")

        sample_conditions = build_sample_condition_table(
            site_group=site_group,
            thresholds=thresholds,
            overlap_df=overlap_df
        )

        sample_conditions.to_csv(
            output_file,
            mode="a",
            index=False,
            header=not header_written
        )

        header_written = True

        del sample_conditions
        gc.collect()

    print(f"Saved {output_file}")


if __name__ == "__main__":
    main()