library(tidyverse)
library(ggplot2)


setwd("PATH/TO/WORKING/DIRECTORY")

df <- read_csv("processed_data/10x/combined_masks_etc_for_upset.csv")

# Total background: full SARS-CoV-2 genome length
N <- 29903

# Select mask columns per primer version (exclude "_prevalent20pct" and "_SB" masks) ---
mask_cols_v3  <- names(df) %>%
  str_subset(".*_ARTIC_3$") %>%
  setdiff(c("ARTIC_3", names(df) %>% str_subset(".*_ARTIC_3_(prevalent20pct|SB)$")))

mask_cols_v4  <- names(df) %>%
  str_subset(".*_ARTIC_4$") %>%
  setdiff(c("ARTIC_4", names(df) %>% str_subset(".*_ARTIC_4_(prevalent20pct|SB)$")))

mask_cols_v41 <- names(df) %>%
  str_subset(".*_ARTIC_4\\.1$") %>%
  setdiff(c("ARTIC_4.1", names(df) %>% str_subset(".*_ARTIC_4\\.1_(prevalent20pct|SB)$")))

# Maybe move NORW from ARTIC 4.1 to ARTIC 4 (already tested this, but whether NORW is included in ARTIC4 or 4.1 there is no significance)
#move_to_v4 <- names(df) %>% str_subset("^NORW_ARTIC_4\\.1$")
#mask_cols_v4 <- union(mask_cols_v4,  move_to_v4)
#mask_cols_v41 <- setdiff(mask_cols_v41, move_to_v4)


# Function to get union of unique, non-NA positions across selected columns
get_union <- function(df, cols) {
  if (length(cols) == 0) return(integer(0))
  v <- df %>% select(all_of(cols)) %>% as.matrix() %>% as.vector()
  unique(v[!is.na(v)])
}

# Union of masked bases per primer version
U_v3 <- get_union(df, mask_cols_v3)
U_v4 <- get_union(df, mask_cols_v4)
U_v41 <- get_union(df, mask_cols_v41)

# Primer base positions
P_v3 <- unique(df$ARTIC_3)
P_v3 <- P_v3[!is.na(P_v3)]
P_v4 <- unique(df$ARTIC_4)
P_v4 <- P_v4[!is.na(P_v4)]
P_v41 <- unique(df$ARTIC_4.1)
P_v41 <- P_v41[!is.na(P_v41)]

# Counts and enrichment test (one-sided hypergeometric: P[X >= x])
supp_tab <- tibble(
  Primer = c("ARTIC v3", "ARTIC v4", "ARTIC v4.1"),
  N_background = N,
  K_primer_bases = c(length(P_v3), length(P_v4), length(P_v41)),
  M_masked_bases = c(length(U_v3), length(U_v4), length(U_v41)),
  Overlap_observed = c(length(intersect(U_v3, P_v3)),
                       length(intersect(U_v4, P_v4)),
                       length(intersect(U_v41, P_v41)))
) %>%
  mutate(
    Primer_coverage_frac = K_primer_bases / N_background,
    Expected_overlap = M_masked_bases * Primer_coverage_frac,
    Fold_observed_expected = Overlap_observed / Expected_overlap,
    p_raw = phyper(q = Overlap_observed - 1,
                   m = K_primer_bases, n = N_background - K_primer_bases,
                   k = M_masked_bases, lower.tail = FALSE),
    q_BH = p.adjust(p_raw, method = "BH")
  )

supp_tab
