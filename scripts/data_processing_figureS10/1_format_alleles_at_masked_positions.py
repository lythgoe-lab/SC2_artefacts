# PER DGP, IDENTIFY THE ALLELES AT THE MASKED POSITIONS
# OUTPUT: .txt FILE PER CENTRE_PROTOCOL LISTING THE POSITIONS MASKED POSITIONS + MOST COMMON ALLELE OBSERVED AS THE MINOR VARIANT ACROSS SAMPLES FROM THE CENTRE_PROTOCOL

import pandas as pd
import os
masked_positions = pd.read_csv("/processed_data/10x/combined_masks_etc_for_upset.csv")
centre_protocols = ["NORT_ARTIC_3","NORT_ARTIC_4","NORT_ARTIC_4.1","NORW_ARTIC_4.1","OXON_VeSeq","PHEC_ARTIC_3","SANG_ARTIC_3","SANG_ARTIC_4.1"]

masked_positions = masked_positions[centre_protocols]

for dgp in centre_protocols:
    output_file = f"/processed_data/S10_covspectrum_analysis/{dgp}_artefacts.txt"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    masked_positions_dgp = masked_positions[dgp].dropna().tolist()
    variants = pd.read_csv(f"/processed_data/10x/{dgp}/output_file.csv") 
    variants_at_masked_positions = variants[variants["Position"].isin(masked_positions_dgp)]
    variants_at_masked_positions = variants_at_masked_positions[(variants_at_masked_positions["read_depth"] > 1000)]
    grouped = variants_at_masked_positions.groupby("Position")

    # AT THE MASKED POSITIONS, FIND WHAT THE MOST COMMON MINOR VARIANT IS
    most_common_allele = grouped["minor_allele1"].apply(
        lambda x: x.loc[(x != "gap") & (x.notna())].mode().iloc[0]
        if not x.loc[(x != "gap") & (x.notna())].mode().empty
        else None
    )
    
    # PUT THAT VARIANT WITH THE TXT FILE
    with open(output_file, "w") as f:        
        for position, allele in most_common_allele.items():
            if allele is not None:
                f.write(f"{int(position)}{allele}\n")
