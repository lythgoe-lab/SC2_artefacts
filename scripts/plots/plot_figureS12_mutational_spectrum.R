library(tidyverse)
library(scales)
library(patchwork)
library(grid)

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

BASE_DIR <- "/processed_data/10x/"
OUTPUT_DIR <- "/figures/"
MASK_FILE      <- file.path(BASE_PATH, "final_masks_all_centres.csv")
THRESHOLD_FILE <- file.path(BASE_PATH, "final_maf_thresholds.csv")
OUTPUT_FIGURE_TI_TV_PATH <- file.path(OUTPUT_DIR, "supplemental/figureS12_mutational_spectrum.png")

# =============================================================================
# 2. HELPER FUNCTIONS
# =============================================================================

#' Parse a Python-style list string into a numeric vector of positions.
#' e.g. "[100, 200, 300]" -> c(100, 200, 300)
parse_mask_string <- function(mask_str) {
  if (length(mask_str) == 0 || is.na(mask_str) || mask_str == "[]") {
    return(numeric(0))
  }
  clean_str <- gsub("\\[|\\]| |'|\"", "", mask_str)
  if (clean_str == "") return(numeric(0))
  as.numeric(unlist(strsplit(clean_str, ",")))
}


#' Load and filter variants for one sequencing centre.
#'
#' @param site          Character. Site/centre identifier.
#' @param base_path     Character. Root directory containing per-site folders.
#' @param maf_threshold Numeric. Minimum minor allele frequency to retain.
#' @param artefact_pos  Numeric vector. Genomic positions flagged as artefacts.
#'
#' @return A tibble with columns: Position, Mutation_Type, Status, Site.
#'         Returns NULL if no data file is found.
load_site_variants <- function(site, base_path, maf_threshold, artefact_pos) {
  
  # Locate the matching subdirectory
  site_dirs <- list.dirs(base_path, recursive = FALSE, full.names = TRUE)
  matches   <- site_dirs[grepl(site, basename(site_dirs), fixed = TRUE)]
  
  if (length(matches) == 0) {
    warning(paste("No directory found for site:", site))
    return(NULL)
  }
  if (length(matches) > 1) {
    message(paste("Multiple directories match", site, "— using:", basename(matches[1])))
  }
  
  variant_file <- file.path(matches[1], "output_file.csv")
  if (!file.exists(variant_file)) {
    warning(paste("No output_file.csv found in:", matches[1]))
    return(NULL)
  }
  
  read_csv(
    variant_file,
    show_col_types = FALSE,
    col_select = c(Position, major_allele, minor_allele1, minor_allele1_freq_no_gaps)
  ) %>%
    filter(minor_allele1_freq_no_gaps > maf_threshold) %>%
    mutate(
      Mutation_Type = paste0(major_allele, ">", minor_allele1),
      Status        = if_else(Position %in% artefact_pos, "Artefact", "True"),
      Site          = site
    ) %>%
    filter(grepl("^[ACGT]>[ACGT]$", Mutation_Type)) %>%
    select(Position, major_allele, Mutation_Type, Status, Site)
}


#' Compute per-site summary statistics: contingency counts, denominators,
#' fold enrichment, and standardised residuals from a chi-square test.
#'
#' @param variants Tibble returned by load_site_variants().
#'
#' @return A named list with elements:
#'   $enrichment — tibble with Fold_Enrichment and log_enrich per Mutation_Type
#'   $residuals  — tibble of standardised residuals per Status × Mutation_Type
#'   $freq       — tibble of percentage composition per Status × Mutation_Type
compute_site_stats <- function(variants) {
  
  # --- Contingency table (counts) ---
  contingency_wide <- variants %>%
    count(Status, Mutation_Type) %>%
    pivot_wider(names_from = Mutation_Type, values_from = n, values_fill = 0)
  
  if (nrow(contingency_wide) < 2) return(NULL)
  
  # --- Denominators: number of distinct positions per ref base × status ---
  denominators <- variants %>%
    distinct(Position, major_allele, Status) %>%
    count(Status, major_allele, name = "Total_Available_Sites")
  
  # --- Chi-square test on the contingency matrix ---
  analysis_matrix <- contingency_wide %>%
    column_to_rownames("Status") %>%
    as.matrix()
  
  chi_result <- chisq.test(analysis_matrix, simulate.p.value = TRUE, B = 10000)
  
  # --- Standardised residuals ---
  residuals_df <- as.data.frame(chi_result$stdres) %>%
    rownames_to_column("Status") %>%
    pivot_longer(cols = -Status, names_to = "Mutation_Type", values_to = "Residual")
  
  # --- Fold enrichment (artefact rate / true rate) ---
  enrichment_df <- contingency_wide %>%
    pivot_longer(cols = -Status, names_to = "Mutation_Type", values_to = "Observed_Count") %>%
    mutate(Ref_Base = substr(Mutation_Type, 1, 1)) %>%
    left_join(denominators, by = c("Status", "Ref_Base" = "major_allele")) %>%
    mutate(Mutation_Rate = Observed_Count / Total_Available_Sites) %>%
    select(Status, Mutation_Type, Mutation_Rate) %>%
    pivot_wider(names_from = Status, values_from = Mutation_Rate) %>%
    mutate(
      Fold_Enrichment = Artefact / True,
      log_enrich      = log2(Fold_Enrichment)
    )
  
  # --- Percentage composition per status (for bar charts) ---
  freq_df <- variants %>%
    count(Status, Mutation_Type) %>%
    group_by(Status) %>%
    mutate(Pct = n / sum(n) * 100) %>%
    ungroup()
  
  list(
    enrichment = enrichment_df,
    residuals  = residuals_df,
    freq       = freq_df
  )
}


# =============================================================================
# 3. DATA LOADING & PROCESSING LOOP
# =============================================================================
maf_data  <- read_csv(THRESHOLD_FILE, show_col_types = FALSE)
mask_data <- read_csv(MASK_FILE, show_col_types = FALSE) %>%
  filter(mask != "positions_in_over_20pct_masks")

unique_sites <- unique(maf_data$Site)

all_enrichments <- list()
all_residuals   <- list()
all_freqs       <- list()

for (site in unique_sites) {
  message("Processing: ", site)
  
  tryCatch({
    # Retrieve site-specific MAF threshold and artefact positions
    maf_threshold   <- maf_data %>% filter(Site == site) %>% pull(MAF_analysis)
    mask_string_val <- mask_data %>% filter(Centre_protocol == site) %>% pull(positions)
    artefact_pos    <- parse_mask_string(mask_string_val)
    
    # Load variants and compute stats
    variants <- load_site_variants(site, BASE_PATH, maf_threshold, artefact_pos)
    if (is.null(variants)) next
    
    stats <- compute_site_stats(variants)
    rm(variants)   # Free memory immediately after summarising
    gc()
    
    if (is.null(stats)) {
      message("Insufficient data for stats — skipping: ", site)
      next
    }
    
    all_enrichments[[site]] <- stats$enrichment %>% mutate(Site = site)
    all_residuals[[site]]   <- stats$residuals   %>% mutate(Site = site)
    all_freqs[[site]]       <- stats$freq        %>% mutate(Site = site)
    
  }, error = function(e) {
    message("Error processing site '", site, "': ", e$message)
  })
}

# Combine into single data frames
final_enrichment_df <- bind_rows(all_enrichments)
final_residuals_df  <- bind_rows(all_residuals)
final_freq_df       <- bind_rows(all_freqs)

clean_site <- function(df) mutate(df, Site = gsub("_", " ", Site))

final_enrichment_df <- clean_site(final_enrichment_df)
final_residuals_df  <- clean_site(final_residuals_df)
final_freq_df       <- clean_site(final_freq_df)


# =============================================================================
# 4. PLOTTING FUNCTIONS
# =============================================================================
# New Dark Grey and Bar Colors
COL_GREY     <- "#333333"  # Dark grey for text and spines
COL_TRUE     <- "#6699CC"  # Muted Blue/Steel for True
COL_ARTEFACT <- "#CC6677"  # Muted Rose/Coral for Artefact

# Mutation order for x-axis (alphabetical within Ti and Tv)
# Grouped by Transitions (Ti) then Transversions (Tv)
ORDERED_MUTATIONS <- c(
  "A>G", "G>A", "C>T", "T>C", # Transitions 
  "A>C", "A>T", "C>G", "C>A", "G>C", "G>T", "T>A", "T>G" # Transversions
)

# For the separator line (positioned between the 4th and 5th items)
TI_TV_SEP <- 4.5

# Shared Nature/Science base theme
theme_custom <- function(base_size = 7) {
  theme_classic(base_size = base_size) +
    theme(
      text             = element_text(family = "sans", color = COL_GREY),
      # 1) Increased Y-axis text size (from base_size to +1)
      axis.text.y      = element_text(size = base_size + 1, color = COL_GREY), 
      axis.text.x      = element_text(size = base_size, color = COL_GREY, angle = 45, hjust = 1),
      axis.title       = element_text(size = base_size + 1, color = COL_GREY),
      axis.line        = element_line(linewidth = 0.4, color = COL_GREY),
      axis.ticks       = element_line(linewidth = 0.4, color = COL_GREY),
      strip.background = element_blank(),
      strip.text       = element_blank(),
      panel.spacing    = unit(0.3, "lines"),
      # 3) Increased Title size (base_size + 3)
      plot.title       = element_text(size = base_size + 3, face = "bold", color = COL_GREY, hjust = 0.5),
      plot.margin      = margin(5, 5, 5, 5)
    )
}

# Apply identical logic to the Ti/Tv version
make_bar_plot_ti_tv <- function(freq_df, status, fill_col, column_title, show_xlab = FALSE, show_ylab = TRUE) {
  
  plot_data <- freq_df %>%
    filter(Status == status) %>%
    mutate(
      Mutation_Type = factor(Mutation_Type, levels = ORDERED_MUTATIONS),
      Site = factor(Site, levels = unique(freq_df$Site))
    )
  
  first_site <- levels(plot_data$Site)[1]
  arrow_data <- tibble(
    Mutation_Type = factor(c("A>G", "C>T", "G>T"), levels = ORDERED_MUTATIONS),
    Site = first_site, y_start = 120, y_end = 105
  )
  
  ggplot(plot_data, aes(x = Mutation_Type, y = Pct)) +
    geom_vline(xintercept = TI_TV_SEP, color = COL_GREY, linetype = "dashed", linewidth = 0.3, alpha = 0.5) +
    geom_col(fill = fill_col, width = 0.75, color = NA) +
    geom_segment(data = arrow_data, aes(x = Mutation_Type, xend = Mutation_Type, y = y_start, yend = y_end),
                 arrow = arrow(length = unit(0.1, "cm")), inherit.aes = FALSE, 
                 linewidth = 0.5, colour = COL_GREY) +
    facet_grid(Site ~ .) +
    scale_y_continuous(limits = c(0, 125), breaks = c(0, 50, 100), labels = label_number(suffix = "%"), expand = c(0, 0)) +
    labs(
      x = if (show_xlab) "Mutation type (Transitions | Transversions)" else NULL,
      y = if (show_ylab) "% of variants" else NULL,
      title = column_title
    ) +
    theme_custom() +
    theme(plot.margin = if(show_ylab) margin(5, 5, 5, 5) else margin(5, 5, 5, 0)) +
    coord_cartesian(clip = "off")
}

#' Build a slim plot of site name labels, one row per site, aligned to the
#' bar chart facet rows. Prepended as col 0 in the patchwork panel.
#'
#' @param site_levels Character vector of site names in display order.
make_site_labels <- function(site_levels) {
  label_df <- tibble(
    Site  = factor(site_levels, levels = site_levels),
    dummy = 1
  )
  ggplot(label_df, aes(x = 1, y = 0.5, label = Site)) +
    geom_text(hjust = 1, size = 8 / .pt, color = "black", fontface = "bold") +
    facet_grid(Site ~ .) +
    scale_x_continuous(limits = c(-0.05, 1), expand = c(0, 0)) +
    scale_y_continuous(limits = c(0, 1),     expand = c(0, 0)) +
    coord_cartesian(clip = "off") +   # Allow text to render outside panel boundary
    theme_void() +
    theme(
      strip.text       = element_blank(),
      strip.background = element_blank(),
      panel.spacing.y  = unit(0.2, "lines"),
      plot.margin      = margin(0, 2, 0, 30)   # Left margin gives text breathing room
    )
}

# =============================================================================
# 5. ASSEMBLE & SAVE PANEL FIGURE
# =============================================================================
site_levels <- unique(final_freq_df$Site)
p_labels <- make_site_labels(site_levels)

p_true_ti_tv     <- make_bar_plot_ti_tv(final_freq_df, "True", COL_TRUE, 
                                        column_title = "True variants", 
                                        show_xlab = FALSE,
                                        show_ylab = TRUE) # Show for Col 1

p_artefact_ti_tv  <- make_bar_plot_ti_tv(final_freq_df, "Artefact", COL_ARTEFACT, 
                                         column_title = "Artefactual variants", 
                                         show_xlab = FALSE,
                                         show_ylab = FALSE) # Suppress for Col 2

panel_figure_ti_tv <- (p_labels + p_true_ti_tv + p_artefact_ti_tv) + 
  plot_layout(ncol = 3, widths = c(0.2, 1, 1)) +
  plot_annotation(
    caption = "Mutation type",
    theme = theme(
      plot.caption = element_text(hjust = 0.6, size = 8, color = COL_GREY, face = "bold")
    )
  )

panel_figure_ti_tv

ggsave(
  filename = OUTPUT_FIGURE_TI_TV_PATH,
  plot     = panel_figure_ti_tv,
  width    = 160, # Adjusted width
  height   = 25 * length(site_levels),
  units    = "mm",
  dpi      = 300,
  bg       = "white"
)

message("Figure saved to: ", OUTPUT_FIGURE_TI_TV_PATH)
