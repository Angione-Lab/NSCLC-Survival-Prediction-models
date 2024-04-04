library(matrixStats)
library(tidyr)
library(ggplot2)
library(dplyr)
library(purrr)
library(ggpubr)
library(reshape2)

#KEGG Pathway for H-VAE-Cox model
#pathway <- read.csv(file = 'Results\\Biological_interpretation\\H_vae_kegg_pathway_barplot_list.csv')

#KEGG Pathway for X-VAE-Cox model
pathway <- read.csv(file = 'Results\\Biological_interpretation\\X-VAE_cox model_kegg_barplot_list.csv')

pathway %>%
  ggplot(aes(x = reorder(ï..Pathways, -Count), y = Count, fill = Benjamini)) +
  geom_bar(stat = "identity") +  
  theme_minimal() + 
  xlab("") + ylab("Top KEGG Pathways ") +
  theme(panel.border = element_blank(), panel.grid.major = element_blank(),
        axis.ticks = element_line(),
        text = element_text(size=14),
        axis.text.x = element_text(hjust = 1, angle = 90, size = 5),
        panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"))


ggsave("Results\\Biological_interpretation\\x_vae_bar_plot.pdf")
