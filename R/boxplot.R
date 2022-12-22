library(magrittr) 
library(dplyr)    
library(tidyverse)
library(ggpubr)
library(rstatix)
library(ggprism)

c_index_results = 'C:/1.workingDirectory/NSCLCSruvivalPrediction/Results/c-index-result.csv'
c_score <- read.table(c_index_results,sep=',', header=TRUE)

pwc <- c_score %>%
  pairwise_t_test(
    Cindex ~ Data, paired = TRUE,
    p.adjust.method = "bonferroni"
  )
pwc


pwc <- pwc %>% add_xy_position(x = "Data")
ggboxplot(c_score, x = "Data", y = "Cindex", add = "point") +
  geom_boxplot(aes(fill = Data), colour = "black") +
  theme(axis.text.x = element_text(angle = 30, hjust = 1, vjust = 0.5)) +
  stat_pvalue_manual(pwc)


ggsave("C:\\1.workingDirectory\\NSCLCSruvivalPrediction\\Results\\boxplot_pval.pdf",height=16, width=25, units="in")

