"""
This script uses a subset of existing VCF files created by LoFreq to identify positions with significant strand bias across samples from the same centre_protocol. 
The main steps are:
1. Index VCF files and load sample metadata.
2. For each eligible centre_protocol (with >=100 samples), randomly subsample 100 samples.
3. For each sample, parse the VCF to extract major and minor allele counts on forward and reverse strands at each position.
4. Aggregate counts across the subsampled samples for each position.
5. Perform Fisher's exact test for strand bias at each position and apply multiple testing correction (FDR).
6. Flag positions that are underpowered based on various criteria (low sample count, low read count, zero cells, low expected counts, pseudo-count sensitivity).
7. Write per-centre results including flags, and create summary CSVs listing positions that pass significance thresholds without being underpowered.
"""

import os
import random
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests


base_directory = '../'
sample_protocol_list = os.path.join(base_directory, "sample_protocol_folder_list.csv")
vcf_files_directory = os.path.join(base_directory, "vcf_files")
output_directory = os.path.join(base_directory, "processed_data", "10x", "strandbias", "100_samples_min10")
os.makedirs(output_directory, exist_ok=True)

VCF_SUFFIX = ".climb_vars.vcf"

# Grouping and sampling
MIN_PROTOCOL_SAMPLES = 100
SUBSAMPLE_SIZE = 100

# Position-level inclusion and flags
MIN_SAMPLES_PER_POS = 10
MIN_TOTAL_READS = 1000
MIN_EXPECTED = 5
PSEUDO_COUNT = 0.5

# Genome trim to coding region
TRIM_START = 265
TRIM_END = 29674

# Significance settings
USE_FDR = True
Q_THRESHOLDS = [0.005, 0.01, 0.05]
P_THRESHOLDS = [0.005, 0.01, 0.05]


def index_vcfs(vcf_dir, suffix):
    m = {}
    dups = []
    for fn in os.listdir(vcf_dir):
        if fn.endswith(suffix):
            name = fn[:-len(suffix)]
            full = os.path.join(vcf_dir, fn)
            if name in m:
                dups.append((name, m[name], full))
            else:
                m[name] = full
    if dups:
        for name, kept, dropped in dups:
            print(f"WARNING: duplicate VCF for sample {name}. Keeping {kept}, ignoring {dropped}")
    return m

def fisher_p(a,b,c,d):
    total = a+b+c+d
    if total == 0:
        return np.nan
    _, p = fisher_exact([[a,b],[c,d]], alternative="two-sided")
    return p

def centre_info_flags(a, b, c, d, sample_count):
    total = a + b + c + d
    # individual flags
    low_sample_flag = sample_count < MIN_SAMPLES_PER_POS
    low_reads_flag  = total < MIN_TOTAL_READS
    zero_cell_flag  = (a == 0) or (b == 0) or (c == 0) or (d == 0)
    # expected counts
    row_tot = [a + b, c + d]
    col_tot = [a + c, b + d]
    grand = sum(row_tot)
    if grand == 0 or (row_tot[0] == 0 and row_tot[1] == 0) or (col_tot[0] == 0 and col_tot[1] == 0):
        low_expected_flag = True
    else:
        exp = [
            row_tot[0] * col_tot[0] / grand, row_tot[0] * col_tot[1] / grand,
            row_tot[1] * col_tot[0] / grand, row_tot[1] * col_tot[1] / grand
        ]
        low_expected_flag = any(e < MIN_EXPECTED for e in exp)
    # pseudo-count sensitivity
    p  = fisher_p(a, b, c, d)
    pp = fisher_p(a + PSEUDO_COUNT, b + PSEUDO_COUNT, c + PSEUDO_COUNT, d + PSEUDO_COUNT)
    pseudo_count_flag = (np.isfinite(p) and np.isfinite(pp) and abs(p - pp) > 0.1)
    # Underpowered if ANY flag is true
    underpowered_flag = (
        low_sample_flag or low_reads_flag or zero_cell_flag or low_expected_flag or pseudo_count_flag
    )
    return dict(
        low_sample_flag=low_sample_flag,
        low_reads_flag=low_reads_flag,
        zero_cell_flag=zero_cell_flag,
        low_expected_flag=low_expected_flag,
        pseudo_count_flag=pseudo_count_flag,
        underpowered_flag=underpowered_flag
    )


def parse_vcf_major_minor_counts(vcf_path):
    per_pos_rows = defaultdict(list)
    with open(vcf_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 8:
                continue
            pos = int(cols[1])
            info = cols[7]
            af = None
            dp4 = None
            for field in info.split(";"):
                if field.startswith("AF="):
                    try:
                        af = float(field.split("=")[1])
                    except:
                        af = None
                elif field.startswith("DP4="):
                    try:
                        dp4 = list(map(int, field.split("=")[1].split(",")))
                    except:
                        dp4 = None
            if dp4 is None or len(dp4) != 4:
                continue
            ref_fw, ref_rv, alt_fw, alt_rv = dp4
            if af is not None and af == 1.0:
                # ensure no REF counts if ALT is consensus with AF=1.0
                ref_fw, ref_rv = 0, 0
            per_pos_rows[pos].append((af if af is not None else 0.0, ref_fw, ref_rv, alt_fw, alt_rv))

    per_pos_counts = {}
    for pos, rows in per_pos_rows.items():
        max_af = max(r[0] for r in rows)
        if max_af > 0.5:
            af, rfw, rrv, afw, arv = max(rows, key=lambda x: x[0])  # consensus ALT (highest AF)
            ref_reads = rfw + rrv
            others = [r for r in rows if r != (af, rfw, rrv, afw, arv) and r[0] < 0.5]
            if others:
                best_other = max(others, key=lambda x: x[3] + x[4])  # strongest <0.5 ALT by reads
                other_reads = best_other[3] + best_other[4]
                if ref_reads >= other_reads:
                    a,b = afw, arv   # major = ALT
                    c,d = rfw, rrv   # minor = REF
                else:
                    a,b = afw, arv   # major = ALT
                    c,d = best_other[3], best_other[4]  # minor = other ALT
            else:
                a,b = afw, arv
                c,d = rfw, rrv
        else:
            af, rfw, rrv, afw, arv = max(rows, key=lambda x: x[0])  # consensus = REF
            a,b = rfw, rrv
            c,d = afw, arv
        per_pos_counts[pos] = (a,b,c,d)
    return per_pos_counts



# Load inputs / index VCFs
samples = pd.read_csv(sample_protocol_list)
vcf_map = index_vcfs(vcf_files_directory, VCF_SUFFIX)
samples = samples[samples["sample_name"].isin(vcf_map.keys())].copy()

# Eligible centres
counts = samples.groupby("centre_protocol")["sample_name"].nunique()
eligible_centres = [cp for cp, n in counts.items() if n >= MIN_PROTOCOL_SAMPLES]

# Get potential mask schemes by p- or q-value thresholds
mask_rows_by_q = {qthr: [] for qthr in Q_THRESHOLDS}
mask_rows_by_p = {pthr: [] for pthr in P_THRESHOLDS}

for centre_protocol in sorted(eligible_centres):
    pool = sorted(samples.loc[samples["centre_protocol"] == centre_protocol, "sample_name"].unique())
    chosen_n = min(SUBSAMPLE_SIZE, len(pool))
    chosen = sorted(random.sample(pool, chosen_n))
    print(f"{centre_protocol}: total={len(pool)}; chosen={len(chosen)}")

    # Aggregates
    sum_counts = defaultdict(lambda: np.array([0,0,0,0], dtype=np.int64))
    sample_count = defaultdict(int)

    for sample_name in chosen:
        per_pos = parse_vcf_major_minor_counts(vcf_map[sample_name])
        for pos, (a,b,c,d) in per_pos.items():
            if pos < TRIM_START or pos > TRIM_END:
                continue
            sum_counts[pos] += np.array([a,b,c,d], dtype=np.int64)
            sample_count[pos] += 1

    # Build per-centre table including all positions and flags
    rows = []
    for pos, vec in sum_counts.items():
        a,b,c,d = map(int, vec.tolist())
        n = sample_count.get(pos, 0)
        p_comb = fisher_p(a,b,c,d)
        flags = centre_info_flags(a,b,c,d, n)
        rows.append({
            "centre_protocol": centre_protocol,
            "position": int(pos),
            "a_major_fw": a, "b_major_rv": b, "c_minor_fw": c, "d_minor_rv": d,
            "sample_count": int(n),
            "total_reads": int(a+b+c+d),
            "p_combined": p_comb,
            **flags
        })

    df = pd.DataFrame(rows)

    # FDR correction
    if len(df):
        finite_mask = df["p_combined"].apply(np.isfinite)
        if USE_FDR and finite_mask.any():
            pvals = df.loc[finite_mask, "p_combined"].values
            _, qvals, _, _ = multipletests(pvals, method="fdr_bh")
            df["q_value"] = np.nan
            df.loc[finite_mask, "q_value"] = qvals
        else:
            df["q_value"] = np.nan

        # Write per-centre results
        out_path = os.path.join(output_directory, f"strandbias_{centre_protocol}.csv")
        df.to_csv(out_path, index=False)
        print(f"  -> wrote {len(df)} rows to {out_path}")

        # Summaries: q-controlled masks
        for qthr in Q_THRESHOLDS:
            sel = df[(df["q_value"].notna()) &
                     (df["q_value"] < qthr) &
                     (~df["underpowered_flag"])]
            positions = sorted(sel["position"].astype(int).tolist())
            mask_rows_by_q[qthr].append({"centre_protocol": centre_protocol, "positions": positions})

        # Summaries: raw p-value masks
        for pthr in P_THRESHOLDS:
            sel = df[(df["p_combined"].notna()) &
                     (df["p_combined"] < pthr) &
                     (~df["underpowered_flag"])]
            positions = sorted(sel["position"].astype(int).tolist())
            mask_rows_by_p[pthr].append({"centre_protocol": centre_protocol, "positions": positions})
    else:
        print(f"  -> no positions observed for centre {centre_protocol}")

# Write summaries to csv
# q-value summaries
for qthr in Q_THRESHOLDS:
    rows = mask_rows_by_q[qthr]
    if not rows:
        continue
    summ = pd.DataFrame(rows)
    summ["positions"] = summ["positions"].apply(lambda x: json.dumps(list(map(int, x))))
    out_name = f"mask_summary_q{str(qthr).replace('0.', '0_')}.csv"
    summ_out = os.path.join(output_directory, out_name)
    summ.to_csv(summ_out, index=False)
    print(f"Wrote mask summary at q<{qthr} to {summ_out}")

# raw p-value summaries
for pthr in P_THRESHOLDS:
    rows = mask_rows_by_p[pthr]
    if not rows:
        continue
    summ = pd.DataFrame(rows)
    summ["positions"] = summ["positions"].apply(lambda x: json.dumps(list(map(int, x))))
    out_name = f"mask_summary_p{str(pthr).replace('0.', '0_')}.csv"
    summ_out = os.path.join(output_directory, out_name)
    summ.to_csv(summ_out, index=False)
    print(f"Wrote mask summary at p<{pthr} to {summ_out}")
