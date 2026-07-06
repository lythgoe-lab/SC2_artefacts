# Code for "Identification and Masking of Artefactual and Misleading Within-Host Variants in Deep-Sequencing SARS-CoV-2 Data"
_Klara M Anker, Rosario Evans Pena, Steven A. Kemp, Joseph Clarke, Lele Zhao, David Bonsall, Nicholas Grayson, Matthew Bashton, The COVID-19 Genomics UK (COG-UK) Consortium, Ann Sarah Walker, Tanya Golubchik, Matthew Hall, Katrina Lythgoe_
___
## Overview
This repository contains the analysis scripts and example data files used for the study _"Identification and Masking of Artefactual and Misleading Within-Host Variants in Deep-Sequencing SARS-CoV-2 Data"_

The study analysed data from the ONS COVID-19 Infection Survey (ONS-CIS) and all sequences are publicly available via the COG-UK project on the ENA, https://www.ebi.ac.uk/ena/browser/view/PRJEB37886 

This repository includes synthetic example data illustrating the folder structure and file formats required to run the analysis pipeline. No real sequencing data are included.

## Repository structure
The repository is organised to reflect the analytical workflow used in the study.

- `scripts/`  
  Contains the processing and plotting scripts used throughout the analysis.  
  Core scripts are numbered to indicate their order within the pipeline.

  Within `scripts/`:
  - `figure4_transmission_pair_examples/` contains scripts used to generate the example transmission pair analyses shown in Figure 4.
  - `plots/` contains all figure-generation scripts used in the manuscript.

- `basefreqs_by_site_filtered/`  
  Illustrates the expected input structure of per-sample base frequency files, organised by sequencing site and coverage threshold.

- `processed_data/`  
  Contains example processed outputs and intermediate files used by downstream scripts, including final masking sets and minor allele frequency (MAF) threshold files used in the analysis.
