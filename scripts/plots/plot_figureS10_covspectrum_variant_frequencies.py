library(tidyverse)
library(ggplot2)
library(patchwork)
library(ggrepel)
library(scales)
library(svglite)

# --- Configuration ---
data_dir <- "/processed_data/S10_covspectrum_analysis/"
sample_count_by_site <- read_csv("/processed_data/S7_temporal_analysis/dummy_weekly_sample_count_per_site.csv")

custom_colors <- c("#FFA69E", "#AA4465", "#721121", "#BB4430", "#995FA3", "#9A98B5", 
                   "#885053", "#777DA7", "#9999C3", "#596F62", "#632A50", "#227C9D", 
                   "#8D6B94", "#B7AAB5", "#335145", "#29335C", "#C08497", "#B56576", 
                   "#065143", "#840032", "#3E4E50", "#749A74", "#4D2D52", "#F1D302", 
                   "#464D77", "#F9DB6D", "#00487C", "#F4D06F", "#395E66", "#F7D6E0", 
                   "#FC7A57", "#F7B801", "#44AF69", "#C2F261", "#373F51", "#6C809A", 
                   "#D35B4C", "#D3A165", "#829CBC", "#37505C", "#DF9A57", "#DB7267", 
                   "#454B66", "#D8DCFF", "#BBACC1", "#00916E", "#88d18a", "#7281A2", 
                   "#d05353", "#f4ac45", "#90c2e7", "#F9DB6D", "#F7C7DB")

dgp_ids = c("NORT_ARTIC_3","NORT_ARTIC_4","NORT_ARTIC_4.1","NORW_ARTIC_4.1","OXON_VeSeq","PHEC_ARTIC_3","SANG_ARTIC_3","SANG_ARTIC_4.1")

# Dictionary mapping dgp_id to proper names
dgp_names <- c("NORT_ARTIC_3" = "NORT, Artic 3",  
               "NORT_ARTIC_4" = "NORT, Artic 4",  
               "NORT_ARTIC_4.1" = "NORT, Artic 4.1",
               "NORW_ARTIC_4.1" = "NORW, Artic 4.1",
               "OXON_VeSeq" = "OXON, VeSEQ",
               "PHEC_ARTIC_3" = "PHEC, Artic 3",
               "SANG_ARTIC_3" = "SANG, Artic 3", 
               "SANG_ARTIC_4.1" = "SANG, Artic 4.1")

# Stat and end dates for the x axis
start_date = "2020-04-27"
end_date = "2023-03-13"

# Compute first and last collection date per dgp_id
date_ranges <- sample_count_by_site %>%
  filter(n_samples > 0) %>%
  group_by(Site) %>%
  summarise(
    start_date = min(collection_week, na.rm = TRUE),
    end_date   = max(collection_week, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  rename(dgp_id = Site) 

# Read and process total sample counts ---
total_counts_path <- file.path(data_dir, "total_sample_counts.csv")
total_counts_df <- read_csv(total_counts_path) %>%
  rename(date = collection_date, total_count = total_count) %>%
  mutate(date = as.Date(date))

# Read and combine all mutation data frames ---
mutation_data_list <- list()

for (dgp_id in dgp_ids) {
  file_path <- file.path(data_dir, dgp_id, paste0("consensus_isnv_variants_fetch_global_", dgp_id, ".csv"))
  
  if (file.exists(file_path)) {
    temp_df <- read_csv(file_path) %>%
      mutate(dgp_id = dgp_id,
             date = as.Date(date)) %>%
      rename(mutation_count = count) # Rename 'count' to 'mutation_count' for clarity
    
    print(temp_df)
    mutation_data_list[[paste0("dgp_", dgp_id)]] <- temp_df
  } else {
    message(paste("Warning: File not found for DGP ID", dgp_id, "-", file_path))
  }
}

# Combine all individual mutation data frames into one large data frame
all_mutations_df <- bind_rows(mutation_data_list)

joined_df <- all_mutations_df %>%
  left_join(total_counts_df, by = "date")

# Calculate the daily proportion for each mutation ---
# The proportion is the count of the specific mutation divided by the total sample count for that day.
final_proportions_df <- joined_df %>%
  mutate(
    mutation_count = as.numeric(mutation_count),
    total_count = as.numeric(total_count),
    proportion = case_when( 
      total_count == 0 ~ 0,
      TRUE ~ mutation_count / total_count
    ) # to avoid division by zero
  ) %>%
  select(dgp_id, date, mutation, mutation_count, total_count, proportion) %>%
  arrange(dgp_id, date, mutation)

# Identify variants that reach >50% frequency ONLY during site-specific dates
variants_above_50 <- final_proportions_df %>%
  left_join(date_ranges, by = "dgp_id") %>%
  filter(date >= start_date & date <= end_date) %>%
  group_by(dgp_id, mutation) %>%
  summarise(max_prop = max(proportion, na.rm = TRUE), .groups = "drop") %>%
  filter(max_prop > 0.5)

# Add labels for variants that reach >50%
all_data <- final_proportions_df %>%
  left_join(variants_above_50,
            by = c("dgp_id", "mutation")) %>%
  mutate(needs_label = !is.na(max_prop))

# Get the peak point for each variant that needs labeling (for label positioning)
label_data <- all_data %>%
  filter(needs_label == TRUE) %>%
  filter(!is.na(proportion) & !is.na(date)) %>%  # Remove NA values
  group_by(dgp_id, mutation) %>%
  slice_max(proportion, n = 1, with_ties = FALSE) %>%  # Get the peak instead of last date
  ungroup()

# Debug: Check if we have any variants to label
print("Variants that reach >50%:")
print(variants_above_50)
print("Number of labels to show:")
print(nrow(label_data))

# Count unique variants per dgp_id for titles
variant_counts <- all_data %>%
  group_by(dgp_id) %>%
  summarise(n_variants = n_distinct(mutation), .groups = "drop")

# Create custom labeller with variant counts using proper names
create_labeller <- function(variant_counts, dgp_names) {
  function(x) {
    counts <- variant_counts$n_variants[match(x, variant_counts$dgp_id)]
    proper_names <- dgp_names[x]
    paste0(proper_names, "(", counts,")")
  }
}

# All variants with labels for those >50%
desired_order <- c("NORT_ARTIC_3","NORT_ARTIC_4","NORT_ARTIC_4.1","NORW_ARTIC_4.1","OXON_VeSeq","PHEC_ARTIC_3","SANG_ARTIC_3","SANG_ARTIC_4.1")
all_data$dgp_id <- factor(all_data$dgp_id, levels = desired_order)
label_data$dgp_id <- factor(label_data$dgp_id, levels = desired_order)
date_ranges$dgp_id <- factor(date_ranges$dgp_id, levels = desired_order) 

label_data_threshold <- all_data %>%
  # Only consider mutations identified in the step above
  filter(needs_label == TRUE) %>%
  # Re-join date ranges to filter the labeling point itself if desired
  left_join(date_ranges, by = "dgp_id") %>%
  filter(date >= start_date & date <= end_date) %>%
  # Pick the peak or the most recent point > 50% within that window
  group_by(dgp_id, mutation) %>%
  slice_max(proportion, n = 1, with_ties = FALSE) %>%
  ungroup()

# Ensure the label data has valid proportion values
label_data_cleaned <- label_data %>%
  filter(proportion >= 0 & proportion <= 1)

# --- PLOT ---
plot1 <- ggplot(all_data, aes(x = date, y = proportion, group = mutation, color = mutation)) +
  # Add the grey rectangle layer for the date ranges
  geom_rect(
    data = date_ranges,
    aes(xmin = start_date, xmax = end_date, ymin = -Inf, ymax = Inf),
    fill = "grey",
    alpha = 0.3,
    inherit.aes = FALSE
  ) +
  geom_line() +
  geom_text_repel(
    data = label_data_threshold,
    aes(x = date, y = proportion, label = mutation),
    size = 5,
    fontface = "bold",
    bg.r = 0.15,
    nudge_y = -0.5,  # Offset the labels
    nudge_x = 2, # Separate overlapping labels
    box.padding = 0.3,
    point.padding = 0.2,
    force = 2,
    max.overlaps = Inf,
    show.legend = FALSE
  ) +
  facet_wrap(
    ~ dgp_id,
    labeller = labeller(dgp_id = create_labeller(variant_counts, dgp_names))
  ) +
  scale_x_date(
    date_breaks = "6 months",
    labels = date_format("%b\n%y"),
    expand = c(0,0),
    date_minor_breaks = "1 month"
  ) +
  scale_y_continuous(limits = c(0,1), expand = c(0,0)) +
  scale_color_manual(values = rep(custom_colors, length.out = length(unique(all_data$mutation)))) +
  theme_minimal() +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major = element_blank(),
    strip.text = element_text(size=13, color = 'black'),
    panel.background = element_rect(fill = "white"),
    plot.background = element_rect(fill = "white"),
    legend.position = "none",
    axis.text.x = element_text(angle = 0, vjust = 0.5, hjust = 1, color = 'black', size = 13),
    axis.text.y = element_text(color = 'black', size = 13),
    axis.ticks = element_line(color = "black"),
    axis.ticks.length = unit(3, "pt"),
    panel.border = element_blank()
  ) +
  labs(
    x = "Date",
    y = "Proportion of GISAID samples\nwith this variant at consensus", size = 13
  )

# print(plot1)

ggsave(filename = "figureS10_variants_at_global_consensus",
       path = "figures/supplemental",
       plot = plot1,
       units = "in",
       width = 10,
       height = 5,
       dpi = 300,
       device = "svg",
       bg = "white")

ggsave(filename = "figureS10_variants_at_global_consensus",
       path = "figures/supplemental",
       plot = plot1,
       units = "in",
       width = 10,
       height = 7,
       dpi = 300,
       device = "png",
       bg = "white")
