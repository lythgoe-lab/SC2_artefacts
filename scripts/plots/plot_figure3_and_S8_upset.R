library(tidyverse)
library(UpSetR)
library(Cairo)


setwd("/Users/kanker/VirusEvolution Dropbox/Klara Marie Anker/ONS_sequencing_restricted_access_shared/Klara_stuff/artefactual_sites_paper/")

data <- read_csv("processed_data/10x/combined_masks_etc_for_upset.csv")

# Transform data to a long format
long_data <- data %>%
  pivot_longer(cols = everything(), names_to = "site", values_to = "position") %>%
  drop_na()

# Create a binary incidence matrix
binary_data <- long_data %>%
  mutate(present = 1) %>%
  distinct() %>%
  pivot_wider(names_from = site, values_from = present, values_fill = list(present = 0)) %>%
  group_by(position) %>%
  summarise(across(everything(), sum)) %>%
  ungroup()


# List of sites to include in Figure 3
sites_fig3 <- list(rev(
  c("DeMaio_Mask", "OXON_VeSeq", "NORT_ARTIC_3", "NORT_ARTIC_4", "NORT_ARTIC_4.1", "NORW_ARTIC_4.1", 
    "PHEC_ARTIC_3", "SANG_ARTIC_3", "SANG_ARTIC_4.1")))


# Function to create and display UpSet plots
create_upset_plot <- function(sites_fig3) {
  # Prepare the input for UpSet plot
  list_input <- binary_data[sites_fig3]
  # Convert to list of positions
  list_input <- lapply(list_input, function(x) which(x == 1))
  # Rename some of the sites with "pretty" names for the final figure
  names(list_input) <- sub("^OXON VeSeq", "OXON ve-SEQ",
                           sub("^NORW ARTIC 4\\.1\\b", "NORW ARTIC ?",
                           sub("^DeMaio", "De Maio",
                               gsub("_", " ", names(list_input)))))
  # Create the UpSet plot
  plot <- upset(fromList(list_input),
                nsets = length(sites_fig3),           # number of sets to include
                nintersects = NA,                     # show all intersections
                keep.order = FALSE,                   # let the order be determined by frequency
                matrix.color = "#264653",             
                main.bar.color = "#2a9d8f",
                mainbar.y.label = "Number of Shared positions",
                sets.bar.color = "#f4a261",
                sets.x.label = "Total Number of Positions",
                point.size = 0.5,
                line.size = 0.3,
                mb.ratio = c(0.65, 0.35),             # view ratio in plot between sets and intersections
                show.numbers = "yes",                 # show numbers on top of intersection bars
                text.scale = c(1,1,1,1,1,1.1),        # scale text sizes
                order.by = "freq",                    # order intersections by frequency
                set_size.show = TRUE,                 # show set size bars
                set_size.numbers_size = 8,            # size of numbers on set size bars
                set_size.angles = 0,                  # angle of numbers on set size bars
                set_size.scale_max = 122
  )
  
  #CairoSVG("figures/figure3_Upsetplot.svg", width = 6.8)
  CairoPDF("figures/figure3_Upsetplot.pdf", width = 6.8)
  print(plot)
  dev.off()
  print(plot)
}

for (site in sites_fig3) {
  create_upset_plot(site)
}


##############################################
# Supplementary figures

# prepare lists of sites to create supplementary upset plots
sites_NORT_NORW <- list(rev(
  c("NORT_ARTIC_3_SB", "NORT_ARTIC_3", "NORT_ARTIC_4_SB", "NORT_ARTIC_4", "NORT_ARTIC_4.1_SB", "NORT_ARTIC_4.1", "NORW_ARTIC_4.1_SB", "NORW_ARTIC_4.1",
    "DeMaio_Mask", "DeMaio_Caution", "ARTIC_3", "ARTIC_4", "ARTIC_4.1")))
sites_PHEC <- list(rev(
  c("PHEC_ARTIC_3_SB", "PHEC_ARTIC_3",
    "DeMaio_Mask", "DeMaio_Caution", "ARTIC_3", "ARTIC_4", "ARTIC_4.1")))
sites_SANG <- list(rev(
  c("SANG_ARTIC_3_SB", "SANG_ARTIC_3", "SANG_ARTIC_4.1_SB", "SANG_ARTIC_4.1", 
    "DeMaio_Mask", "DeMaio_Caution", "ARTIC_3", "ARTIC_4", "ARTIC_4.1")))


create_upset_plot_NORT_NORW <- function(sites_NORT_NORW) {
  list_input <- binary_data[sites_NORT_NORW]
  list_input <- lapply(list_input, function(x) which(x == 1))
  
  names(list_input) <- sub("^OXON VeSeq", "OXON ve-SEQ",
                           sub("^NORW ARTIC 4\\.1\\b", "NORW ARTIC ?",
                               sub("^DeMaio", "De Maio",
                                   gsub("_", " ", names(list_input)))))
  
  # Create the UpSet plot with adjusted number angle
  plot <- upset(fromList(list_input),
                nsets = length(sites_NORT_NORW),
                nintersects = NA,
                keep.order = FALSE,
                matrix.color = "#264653",
                main.bar.color = "#2a9d8f",
                mainbar.y.label = "Number of Shared positions",
                mainbar.y.max = 3800,
                sets.bar.color = "#f4a261",
                sets.x.label = "Total Number of Positions",
                point.size = 0.5,
                line.size = 0.3,
                mb.ratio = c(0.5, 0.5),
                show.numbers = "yes", number.angles = 0,
                text.scale = c(1,1,1,1,1,1),
                order.by = "freq",
                set_size.show = TRUE,
                set_size.numbers_size = 8,
                set_size.angles = 30
  )
  
  CairoSVG("figures/supplemental/figureS8_NORT.svg", width = 7.5, height = 4)
  print(plot)
  dev.off()
  print(plot)
}

for (site in sites_NORT_NORW) {
  create_upset_plot_NORT_NORW(site)
}

create_upset_plot_PHEC <- function(sites_PHEC) {
  list_input <- binary_data[sites_PHEC]
  list_input <- lapply(list_input, function(x) which(x == 1))
  
  names(list_input) <- sub("^OXON VeSeq", "OXON ve-SEQ",
                           sub("^NORW ARTIC 4\\.1\\b", "NORW ARTIC ?",
                               sub("^DeMaio", "De Maio",
                                   gsub("_", " ", names(list_input)))))
  
  # Create the UpSet plot with adjusted number angle
  plot <- upset(fromList(list_input),
                nsets = length(sites_PHEC),
                nintersects = NA,
                keep.order = FALSE,
                matrix.color = "#264653",
                main.bar.color = "#2a9d8f",
                mainbar.y.label = "Number of Shared positions",
                mainbar.y.max = 3800,
                sets.bar.color = "#f4a261",
                sets.x.label = "Total Number of Positions",
                point.size = 0.5,
                line.size = 0.3,
                mb.ratio = c(0.5, 0.5),
                show.numbers = "yes", number.angles = 0,
                text.scale = c(1,1,1,1,1,1),
                order.by = "freq",
                set_size.show = TRUE,
                set_size.numbers_size = 8,
                set_size.angles = 30
  )
  
  CairoSVG("figures/supplemental/figureS8_PHEC.svg", width = 7.5, height = 3.5)
  print(plot)
  dev.off()
  print(plot)
}

for (site in sites_PHEC) {
  create_upset_plot_PHEC(site)
}


create_upset_plot_SANG <- function(sites_SANG) {
  list_input <- binary_data[sites_SANG]
  list_input <- lapply(list_input, function(x) which(x == 1))
  
  names(list_input) <- sub("^OXON VeSeq", "OXON ve-SEQ",
                           sub("^NORW ARTIC 4\\.1\\b", "NORW ARTIC ?",
                               sub("^DeMaio", "De Maio",
                                   gsub("_", " ", names(list_input)))))
  
  # Create the UpSet plot with adjusted number angle
  plot <- upset(fromList(list_input),
                nsets = length(sites_SANG),
                nintersects = NA,
                keep.order = FALSE,
                matrix.color = "#264653",
                main.bar.color = "#2a9d8f",
                mainbar.y.label = "Number of Shared positions",
                mainbar.y.max = 3800,
                sets.bar.color = "#f4a261",
                sets.x.label = "Total Number of Positions",
                point.size = 0.5,
                line.size = 0.3,
                mb.ratio = c(0.55, 0.5),
                show.numbers = "yes", number.angles = 0,
                text.scale = c(1,1,1,1,1,1),
                order.by = "freq",
                set_size.show = TRUE,
                set_size.numbers_size = 8,
                set_size.angles = 30
  )
  
  CairoSVG("figures/supplemental/figureS8_SANG.svg", width = 7.5, height = 3.8)
  print(plot)
  dev.off()
  print(plot)
}

for (site in sites_SANG) {
  create_upset_plot_SANG(site)
}





