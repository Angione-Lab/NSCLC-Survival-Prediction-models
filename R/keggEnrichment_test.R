# # https://support.bioconductor.org/p/9150409/
#https://karobben.github.io/2020/06/14/R/clusterProfiler/

#library(DESeq2)
#BiocManager::install("EnsDb.Hsapiens.v86")
library(EnsDb.Hsapiens.v86)
#BiocManager::install("EnhancedVolcano")
#library(EnhancedVolcano)
#BiocManager::install("pheatmap")
library(pheatmap)
#BiocManager::install("org.Hs.eg.db")
library(org.Hs.eg.db)
#BiocManager::install("KEGGREST")
library(KEGGREST)
#BiocManager::install("pathview")
library(pathview)
#BiocManager::install("clusterProfiler")
#library(KEGG.db)
library(clusterProfiler)
#BiocManager::install("gage")
library(gage)
#BiocManager::install("gageData")
library(gageData)
library(R.utils)

library(enrichplot)
library(ggplot2)
R.utils::setOption("clusterProfiler.download.method","auto")

H_vae_shapGene =  read.csv('Results\\Biological_interpretation\\HVAE-GeneShapImportance.csv')
X_vae_shapGenes =  read.csv('Results\\X_VAE_cox\\good results\\att_XVAE-GeneShapImportance5_final.csv')

H_vae_shapGene = head(H_vae_shapGene, 40)
X_vae_shapGenes = head(X_vae_shapGenes, 40)
X_vae_shapGenes <- X_vae_shapGenes %>% select(-c("X"))
colnames(X_vae_shapGenes) <- colnames(H_vae_shapGene)

common_genes = intersect(H_vae_shapGene$features,X_vae_shapGenes$features)


id <- mapIds(org.Hs.eg.db, keys=genes$features,,, column="ENTREZID", keytype="SYMBOL")
a = data.frame(id)
a$Feat_importance <- genes$Importance
a <- a[!duplicated(a[ , c("id")]),]
a<-na.omit(a)
rownames(a) <- a$id
a <- a %>% select(-c("id"))


#upset plot
de <- rownames(a)
edo <- enrichDGN(de)
upsetplot(edo)





# Perform GO enrichment analysis for all three categories
go_bp <- enrichGO(gene = rownames(a), OrgDb = org.Hs.eg.db, ont = "BP", pAdjustMethod = "BH", pvalueCutoff = 0.005)
go_cc <- enrichGO(gene = rownames(a), OrgDb = org.Hs.eg.db, ont = "CC", pAdjustMethod = "BH", pvalueCutoff = 0.005)
go_mf <- enrichGO(gene = rownames(a), OrgDb = org.Hs.eg.db, ont = "MF", pAdjustMethod = "BH", pvalueCutoff = 0.005)

go_bp@result$group = "BP"
go_cc@result$group = "CC"
go_mf@result$group = "MF"

# Summarize the results in a table
go_results <- rbind(summary(go_bp), summary(go_cc), summary(go_mf))

ggplot(go_results, aes(x=Description, y=Count, fill=group)) +
  geom_bar(stat = 'identity') +
  facet_grid(~group, scales = 'free', space = 'free') + 
  theme_bw() +
  theme(axis.text.x = element_text(angle = 270, hjust = 0, vjust = .5), legend.position = 'none', panel.grid =  element_blank(), strip.background = element_rect(fill = 'white'))
#ggsave_GO(paste("GO/", Sample_Dir, File_name, ".png", sep="" ), nrow(GO_TB))




dotplot(go_bp, showCategory = 15)


#%%
de <- rownames(a)
edo <- enrichDGN(de)
edox <- setReadable(edo, 'org.Hs.eg.db', 'ENTREZID')
edox2 <- pairwise_termsim(edox)
p1 <- treeplot(edox2)
p2 <- treeplot(edox2, hclust_method = "average")
aplot::plot_list(p1, p2, tag_levels='A')

#%%

upsetplot(edo)
##
x <- enrichDO(de)
set.seed(2020-10-27)
selected_pathways <- sample(x$Description, 20)
selected_pathways
p2 <- dotplot(x, showCategory = selected_pathways, font.size=14)



edo <- enrichDGN(rownames(a))
barplot(edo, showCategory=30) 
mutate(edo, qscore = -log(p.adjust, base=10)) %>% 
  barplot(x="qscore")


kk <- enrichKEGG(gene         = rownames(a),
                 organism     = 'hsa',
                 #keyType = 'kegg',
                 pvalueCutoff = 0.05,
                 use_internal_data = TRUE)
head(kk)
#barplot(kk, showCategory=30, x = "ID") 
kk@result$Description = kk$ID
b <- data.frame(kk)
b$Description <- b$ID


mkk <- enrichMKEGG(gene = rownames(a),
                   organism = 'hsa',
                   pvalueCutoff = 0.05)
head(mkk)  

ck <- setReadable(mkk, OrgDb = org.Hs.eg.db, keyType="ENTREZID")
head(ck) 
dotplot(ck, showCategory = 20)


enrichplot::dotplot(kk, x = 'ID', color="pvalue")
# https://guangchuangyu.github.io/2016/01/go-analysis-using-clusterprofiler/
organism = 'hsa' 
go_enrich <- enrichGO(gene = rownames(a),
                     # universe = names(a),
                      OrgDb = 'org.Hs.eg.db', 
                      # keyType = 'ENSEMBL',
                      readable = T,
                      ont = "BP",
                      pvalueCutoff = 0.05)
dotplot(go_enrich, showCategory = 20)

david = enrichDAVID(gene = rownames(a), idType="SYMBOL",  annotation="KEGG_PATHWAY")
