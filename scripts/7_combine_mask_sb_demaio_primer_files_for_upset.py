"""
This script combines the final mask sets for each sequencing centre with other lists of positions for comparisons in upset plots. 
The combined file includes:
- The final mask sets for each centre (both adaptive and prevalent)
- positions with significant strand bias from the strand bias analysis
- The DeMaio mask and caution lists
- The positions of the ARTIC v3, v4 and v4.1 primers
"""

import pandas as pd
import json

def bed_to_position_list(bed_path):
    bed = pd.read_csv(bed_path, sep="\t", header=None, usecols=[1, 2], names=["start", "end"])
    positions = set()
    for s, e in zip(bed["start"], bed["end"]):
        positions.update(range(int(s), int(e)))
    return sorted(positions)

def read_demaio_positions(path):
    df = pd.read_csv(path, header=None, names=["position"], dtype=str)
    df = df[df["position"].str.strip().str.lower() != "position"]
    df = df[df["position"].str.strip() != ""]
    return df["position"].astype(int).tolist()

def combine_masks(
    masks_all_centres_path: str,
    strandbias_path: str,
    demaio_mask_path: str,
    demaio_caution_path: str,
    primer_bed_paths: dict,
    output_path: str
) -> None:
    df_masks = pd.read_csv(masks_all_centres_path)
    adaptive_masks = df_masks[df_masks["mask"] != "positions_in_over_20pct_masks"]
    prevalent_masks = df_masks[df_masks["mask"] == "positions_in_over_20pct_masks"]
    df_strandbias = pd.read_csv(strandbias_path)

    masks_dict = {}

    for _, row in adaptive_masks.iterrows():
        masks_dict[row["Centre_protocol"]] = json.loads(row["positions"])

    for _, row in prevalent_masks.iterrows():
        key = f"{row["Centre_protocol"]}_prevalent20pct"
        masks_dict[key] = json.loads(row["positions"])

    for _, row in df_strandbias.iterrows():
        if not row["centre_protocol"].endswith("Unknown"):
            key = f"{row["centre_protocol"]}_SB"
            masks_dict[key] = json.loads(row["positions"])

    masks_dict["DeMaio_Mask"] = read_demaio_positions(demaio_mask_path)
    masks_dict["DeMaio_Caution"] = read_demaio_positions(demaio_caution_path)

    df_masks_expanded = pd.DataFrame({k: pd.Series(v) for k, v in masks_dict.items()})

    primer_lists = {name: bed_to_position_list(path) for name, path in primer_bed_paths.items()}

    n_rows = max(len(df_masks_expanded), *(len(lst) for lst in primer_lists.values()))

    df_masks_expanded = df_masks_expanded.apply(lambda col: col.reindex(range(n_rows)))

    for primer_name, pos_list in primer_lists.items():
        df_masks_expanded[primer_name] = pd.Series(pos_list).reindex(range(n_rows))

    df_masks_expanded.to_csv(output_path, index=False)

if __name__ == "__main__":
    masks_csv = "../processed_data/10x/final_masks_all_centres.csv"
    strandbias_csv = "../processed_data/10x/strandbias/100_samples_min10/mask_summary_q0_01.csv"
    demaio_mask_csv = "../demaio_mask.csv"
    demaio_caution_csv = "../demaio_caution.csv"

    primer_beds = {
        "ARTIC_3": "../ARTIC_3.primer.bed",
        "ARTIC_4": "../ARTIC_4.primer.bed",
        "ARTIC_4.1": "../ARTIC_4.1.primer.bed"
    }

    output_csv = "../processed_data/10x/combined_masks_etc_for_upset.csv"

    try:
        combine_masks(
            masks_all_centres_path=masks_csv,
            strandbias_path=strandbias_csv,
            demaio_mask_path=demaio_mask_csv,
            demaio_caution_path=demaio_caution_csv,
            primer_bed_paths=primer_beds,
            output_path=output_csv
        )
        # Count number of masks combined (excluding primers)
        num_masks = len([
            *pd.read_csv(masks_csv)["Centre_protocol"].unique(),
            "DeMaio_Mask",
            "DeMaio_Caution",
            *[f"{cp}_prevalent20pct" for cp in pd.read_csv(masks_csv)["Centre_protocol"].unique()],
            *[f"{cp}_SB" for cp in pd.read_csv(strandbias_csv)["centre_protocol"].unique() if not cp.endswith("Unknown")]
        ])
        num_primers = len(primer_beds)
        print(f"combine_masks completed successfully. Output written to: {output_csv}")
        print(f"Number of masks in combined file (excluding primers): {num_masks}")
        print(f"Number of primers added: {num_primers}")
    except FileNotFoundError as e:
        print(f"Error: File not found - {e.filename}")
    except pd.errors.EmptyDataError as e:
        print(f"Error: One of the input files is empty - {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
