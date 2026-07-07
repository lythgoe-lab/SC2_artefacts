library(arrow)
library(dplyr)
library(tidyverse)
library(lubridate)
library(ggplot2)
library(tidyr)
library(patchwork)

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
BASE_DIR <- "/processed_data/"
OUTPUT_DIR <- "/figures/"

ONS_SUMMARY_PATH <- "ons_summary.csv"

MASK_FILE_PATH <- file.path(BASE_DIR, "10x/final_masks_all_centres.csv")
THRESHOLD_FILE_PATH <- file.path(BASE_DIR, "10x/final_maf_thresholds.csv")

FINAL_TEMPORAL_SUMMARY_PATH <- file.path(BASE_DIR, "S7_temporal_analysis/temporal_summary.csv")
SAMPLE_COUNTS_LONG_PATH <- file.path(BASE_DIR, "S7_temporal_analysis/weekly_sample_count_per_site.csv")

OUTPUT_PATH = file.path(OUTPUT_DIR, "supplemental/figureS7_temporal_distribution.png")

# CONSTANTS
SITE_FOLDER_MAP <- list(
  "SANG_ARTIC_4.1" = c("BRBR_ARTIC_4.1_ILLUMINA", "LSPA_ARTIC_4.1_ILLUMINA", "QEUH_ARTIC_4.1_ILLUMINA"),
  "SANG_ARTIC_3"   = c("MILK_ARTIC_3_ILLUMINA", "QEUH_ARTIC_3_ILLUMINA"),
  "NORW_ARTIC" = c("NORW_ARTIC_Unknown_ILLUMINA"),
  # 1:1 sites — folder name matches site name (grepl match handles these)
  "NORT_ARTIC_3"   = c("NORT_ARTIC_3_ILLUMINA"),
  "NORT_ARTIC_4"   = c("NORT_ARTIC_4_ILLUMINA"),
  "NORT_ARTIC_4.1" = c("NORT_ARTIC_4.1_ILLUMINA"),
  "OXON_ve-SEQ"     = c("OXON_VeSeq_ILLUMINA"),
  "PHEC_ARTIC_3"   = c("PHEC_ARTIC_3_ILLUMINA")
)

SITE_ORDER <- c(
  "SANG_ARTIC_3", "OXON_ve-SEQ", "PHEC_ARTIC_3", "NORT_ARTIC_3",
  "NORT_ARTIC_4", "NORW_ARTIC", "NORT_ARTIC_4.1", "SANG_ARTIC_4.1"
)

# =============================================================================
# 2. HELPER FUNCTIONS
# =============================================================================
parse_mask_string <- function(mask_str) {
  if (length(mask_str) == 0 || is.na(mask_str) || mask_str == "[]") return(numeric(0))
  clean_str <- gsub("\\[|\\]| |'|\"", "", mask_str)
  if (clean_str == "") return(numeric(0))
  as.numeric(unlist(strsplit(clean_str, ",")))
}

# =============================================================================
# 3. DATA LOADING
# =============================================================================
# --- 1. Load Temporal Analysis Outputs ---
final_temporal_summary <- read.csv(FINAL_TEMPORAL_SUMMARY_PATH)
sample_counts_long <- read.csv(SAMPLE_COUNTS_LONG_PATH)

# --- 2. Load Sequence Metadata ---
ons_summary <- read_csv(ONS_SUMMARY_PATH, show_col_types = FALSE) %>%
  select(sample_name, collection_date)
maf_data <- read_csv(THRESHOLD_FILE_PATH, show_col_types = FALSE)
mask_data <- read_csv(MASK_FILE_PATH, show_col_types = FALSE)

# =============================================================================
# 4. DATA PROCESSING
# =============================================================================
# --- 4.1 Parse Mask Positions ---
# Aggregate ALL positions that were masked in ANY centre/protocol
all_masked_positions <- mask_data %>%
  filter(mask != "positions_in_over_20pct_masks") %>%
  pull(positions)

masked_positions <- unique(unlist(map(all_masked_positions, parse_mask_string)))

thresholds <- unique(maf_data$MAF_analysis)

# --- 4.2. Initialize Global Result Container ---
unique_sites <- unique(maf_data$Site) # get the unique protocols to iterate over

# --- 4.3. Parse mask_data to long format: one row per site/position ---
masked_long <- mask_data %>%
  filter(mask != "positions_in_over_20pct_masks") %>%
  mutate(positions_parsed = map(positions, parse_mask_string)) %>%
  select(Centre_protocol, positions_parsed) %>%
  unnest(positions_parsed) %>%
  dplyr::rename(Position = positions_parsed) %>%
  dplyr::rename(Site = Centre_protocol) %>%
  distinct(Site, Position)

# --- 4.4. Prep temporal data for plot
# --- Reshape to long format, one row per position/week/threshold ---
heatmap_data <- final_temporal_summary %>%
  pivot_longer(
    cols = starts_with("hits_at_"),
    names_to = "threshold",
    values_to = "count"
  ) %>%
  mutate(
    threshold = gsub("hits_at_|_pct", "", threshold),
    threshold = paste0(threshold, "%"),
    pct_with_isnv = (count / total_samples_1000x) * 100
  ) %>%
  inner_join(masked_long, by = "Position") %>%      # duplicates positions appearing in multiple sites
  mutate(
    # change VeSeq -> ve-SEQ & NORW_ARTIC_4.1 -> NORW_ARTIC for consistency with other figures
    Site = case_match(
      Site,
      "OXON_VeSeq"            ~ "OXON_ve-SEQ",
      "NORW_ARTIC_4.1"   ~ "NORW_ARTIC",
      .default           = as.character(Site)
    ),
    Site = factor(Site),
    
    pos_label = paste0(Site, "_", Position),         # unique y-axis unit: site + position
    pos_label = factor(pos_label),
    collection_week = ymd(collection_week)
  )

sample_counts_long <- sample_counts_long %>%
  mutate(collection_week = ymd(collection_week))%>%
  mutate(
    # change VeSeq -> ve-SEQ & NORW_ARTIC_4.1 -> NORW_ARTIC for consistency with other figures
    Site = case_match(
      Site,
      "OXON_VeSeq"            ~ "OXON_ve-SEQ",
      "NORW_ARTIC_4.1"   ~ "NORW_ARTIC",
      .default           = as.character(Site)
    ))

# Set factor level order FIRST — rev() so top of plot = first in site_order
heatmap_data <- heatmap_data %>%
  mutate(pos_label = factor(pos_label, levels = rev(unique(pos_label[order(
    match(Site, rev(site_order)), pos_label
  )]))))

# THEN derive site_labels — y positions reflects the new ordering
site_labels <- heatmap_data %>%
  distinct(Site, pos_label) %>%
  group_by(Site) %>%
  summarise(
    y_mid = median(as.numeric(pos_label)),
    y_min = min(as.numeric(pos_label)),
    y_max = max(as.numeric(pos_label)),
    .groups = "drop"
  )

# =============================================================================
# 5. PLOT
# =============================================================================
# Calculate Shared Limits & Label Positioning
all_dates <- c(sample_counts_long$collection_week, heatmap_data$collection_week)
shared_limits <- range(all_dates, na.rm = TRUE)
label_x_pos <- max(shared_limits) + 20  # Position for text: 20 days after the last data point

# Use the exact same expansion and margin for both plots
right_expansion <- 220 
label_x_pos <- max(shared_limits) + 25 
right_margin <- 5  

# Get the default 'Set2' colors for the remaining 7 sites
# Set2 has 8 colors max; take them all and then override the first one
default_palette <- RColorBrewer::brewer.pal(8, "Set2")

# Create the named vector
site_colors <- setNames(default_palette, site_order)

# Override the first colour one with specific dark blue
site_colors["SANG_ARTIC_3"] <- "#011959"

# Apply the order to the data frame
sample_counts_long <- sample_counts_long %>%
  mutate(Site = factor(Site, levels = site_order))

#### Plot HISTOGRAM
p_hist <- ggplot(sample_counts_long, aes(x = collection_week, y = n_samples, fill = Site)) +
  # Keeps SANG_ARTIC_3 at the bottom of the stack
  geom_col(position = position_stack(reverse = TRUE)) + 
  facet_wrap(~ "Sequencing Depth") + 
  scale_x_date(
    date_breaks = "3 months", 
    date_labels = "%b %y",
    expand = expansion(add = c(0, right_expansion)) 
  ) +
  coord_cartesian(xlim = shared_limits, clip = "off") +
  
  # Apply the custom-mixed palette
  scale_fill_manual(
    values = site_colors,
    name = "Sequencing Method"
  ) +
  
  guides(fill = guide_legend(
    title.theme = element_text(size = 12, color = "black", face = "bold"),
    label.theme = element_text(size = 9.5, color = "black"),
    reverse = FALSE # Keeps legend matching your site_order list
  )) +
  ylab("Number of Sequenced Samples") +
  theme_minimal() +
  theme(
    axis.text.y = element_text(size = 9.5, colour = 'black'),
    axis.title.y = element_text(size = 12, colour = 'black'),
    axis.ticks.y = element_blank(),
    axis.text.x = element_blank(),
    axis.title.x = element_blank(),
    axis.ticks.x = element_blank(),
    strip.background = element_blank(),
    strip.text = element_text(color = "transparent"), 
    plot.margin = margin(5, right_margin, 0, 5),
    panel.grid = element_blank()
  )

# Plot HEATMAP
p_heat <- ggplot(heatmap_data, aes(x = collection_week, y = pos_label, fill = pct_with_isnv)) +
  geom_tile() +
  geom_hline(
    data = site_labels %>% filter(y_min > 1),
    aes(yintercept = y_min - 0.5),
    colour = "grey40", linewidth = 0.4, linetype = "dashed"
  ) +
  geom_text(
    data = site_labels,
    aes(x = label_x_pos, y = y_mid, label = Site),
    inherit.aes = FALSE, hjust = 0, size = 2.8, colour = "grey20", fontface = "bold"
  ) +
  scale_fill_gradientn(
    colours = c("white", "#FEE391", "#FE9929", "#CC4C02"),
    name = "% samples\nwith iSNV"
  ) +
  # --- LEGEND AESTHETICS FOR HEATMAP ---
  guides(fill = guide_colorbar(
    title.theme = element_text(size = 12, color = "black", face = "bold"),
    label.theme = element_text(size = 9.5, color = "black"),
    barwidth = 1,
    barheight = 6
  )) +
  scale_x_date(
    date_breaks = "3 months", 
    date_labels = "%b %y",
    expand = expansion(add = c(0, right_expansion))
  )+
  coord_cartesian(xlim = shared_limits, clip = "off") + 
  # --- CUSTOM LABELLER FOR STRIP ---
  facet_wrap(~ threshold, ncol = 1, 
             labeller = labeller(threshold = function(x) paste("MAF threshold:", x))) +
  ylab("Artefact Positions\n(Grouped by Sequencing Method)")+
  xlab("Sample Collection (Grouped by Week)")+
  theme_minimal()+
  theme(
    axis.text.y = element_blank(),
    axis.title.y = element_text(color = "black", size = 12),
    axis.text.x = element_text(color = "black", size = 9.5),
    axis.title.x = element_text(color = "black", size = 12),
    panel.grid = element_blank(),
    # --- PUSH STRIP TEXT TO LEFT ---
    strip.text = element_text(face = "bold", hjust = 0, size = 10),
    plot.margin = margin(0, right_margin, 5, 5) 
  )

# --- 4. Assembly ---
combined_plot <- (p_hist / p_heat) + 
  plot_layout(heights = c(1, 4), guides = "collect") & 
  theme(
    legend.position = "right",
    legend.justification = "top",
    legend.box.margin = margin(l = -10) 
  )

ggsave(plot = combined_plot, file = OUTPUT_PATH, width = 11, height = 13)


