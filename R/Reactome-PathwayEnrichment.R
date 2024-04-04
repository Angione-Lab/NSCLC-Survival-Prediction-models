
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

BiocManager::install("ReactomePA")
BiocManager::install("org.Hs.eg.db")

library(ReactomePA)
library(org.Hs.eg.db)
library(ggplot2)
library(enrichplot)
library(clusterProfiler)
#Reactome Pathway for X-VAE-Cox model
#file_path = 'Results\\Biological_interpretation\\XVAE-GeneShapImportance.csv'

#Reactome Pathway for H-VAE-Cox model
file_path = 'Results\\Biological_interpretation\\HVAE-GeneShapImportance.csv'
genes =  read.csv(file = file_path)
id <- mapIds(org.Hs.eg.db, keys=genes$feature, column="ENTREZID", keytype="SYMBOL")
a = data.frame(id)
a$mean <- genes$Mean
a <- dplyr::filter(a,  !is.na(id))
rownames(a) <- a$id
significant_genes <- head(a, 1714)

x <- enrichPathway(gene=significant_genes$id, pvalueCutoff = 0.05, readable=TRUE)
#head(x)
dotplot(x, showCategory=30)

#ggsave("Results\\Biological_interpretation\\H_vae_reactome.pdf",height=16, width=15, units="in")



#cnetplot(x, categorySize="pvalue", foldChange=NULL)



