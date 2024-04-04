library(magrittr) 
library(dplyr)    
library(tidyverse)
library(ggpubr)
library(rstatix)
library(ggprism)

c_index_results = 'Results/c-index-result.csv'
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
  theme(axis.text.x = element_text(angle = 30, hjust = 1, vjust = 1)) +
  stat_pvalue_manual(pwc, hide.ns = TRUE)


ggsave("Results\\boxplot_pval.pdf",height=16, width=25, units="in")


a <- c_score %>%select(c(Cindex, Data)) %>% group_by(Data) %>% summarise_each(funs(mean, sd))

compare_means(Cindex ~ Data,  data = c_score,
              ref.group = ".all.", method = "t.test")

ggboxplot(c_score, x = "Data", y = "Cindex", color = "Data", 
          add = "jitter", legend = "none", facet.by = "Dataset",  short.panel.labs = FALSE) +
  rotate_x_text(angle = 45)+
  geom_hline(yintercept = mean(c_score$Cindex), linetype = 2)+ # Add horizontal line at base mean
  # stat_compare_means(method = "anova")+        # Add global annova p-value
  stat_compare_means(label = "p.signif", method = "t.test", label.y = 0.9, ref.group = ".all.", group.by = "Dataset")  


ggsave("Results/updated_boxplot_pval1.pdf",height=8, width=25, units="in")

