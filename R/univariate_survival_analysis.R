library("survival")
library("survminer")
library(dplyr)

#RNA_Seq = 'NSCLC-Survival-Prediction-models/Data/sorted_RNA_Seq.csv'
#RNA_Seq = 'NSCLC-Survival-Prediction-models/downlampled_pathway_normalised_images_simple.csv'
lat = 'Results/cox_image_latent.csv'
clic_data = 'Data/NSCLC_Clinical_Data.csv'


feature_data <- read.table(lat, row.names=1,sep=',', header=TRUE)
clinical_data <- read.table(clic_data, row.names=1,sep=',', header=TRUE)

common <- intersect(rownames(feature_data), rownames(clinical_data))  
feature_data <- feature_data[common,] 
clinical_data <- clinical_data[common,] 
clinical_data <- clinical_data[order(common), ]
feature_data <- feature_data[order(common), ]

clinical_data[clinical_data == 'Alive'] <- 0
clinical_data[clinical_data == 'Dead'] <- 1
clinical_data[, c('Survival.Status')] <- sapply(clinical_data[, c('Survival.Status')], as.numeric)

rna_surv = merge(x=feature_data,y=clinical_data[, c('survival_time', 'Survival.Status')], by.x = 0, by.y = 0)
rownames(rna_surv) <- rna_surv$Row.names
rna_surv<- rna_surv %>% select(-Row.names)

#res.cox <- coxph(Surv(survival_time, Survival.Status) ~ EGFR, data = rna_surv)

#res


covariates <- colnames(feature_data)
univ_formulas <- sapply(covariates,
                        function(x) as.formula(paste('Surv(survival_time, Survival.Status)~', x)))

univ_models <- lapply( univ_formulas, function(x){coxph(x, data = rna_surv)})
# Extract data 
univ_results <- lapply(univ_models,
                       function(x){ 
                         x <- summary(x)
                         p.value<-signif(x$wald["pvalue"])
                         wald.test<-signif(x$wald["test"])
                         beta<-signif(x$coef[1]);#coeficient beta
                         HR <-signif(x$coef[2]);#exp(beta)
                         HR.confint.lower <- signif(x$conf.int[,"lower .95"], 2)
                         HR.confint.upper <- signif(x$conf.int[,"upper .95"],2)
                         HR <- paste0(HR, " (", 
                                      HR.confint.lower, "-", HR.confint.upper, ")")
                         res<-c(beta, HR, wald.test, p.value)
                         names(res)<-c("beta", "HR (95% CI for HR)", "wald.test", 
                                       "p.value")
                         return(res)
                         #return(exp(cbind(coef(x),confint(x))))
                       })
res <- t(as.data.frame(univ_results, check.names = FALSE))
#write.csv(as.data.frame(res), "C:/1.WorkingDrive/Research documents/NSCLC-Survival-Prediction-models/Data/univariate_latent_cox.csv")
a = as.data.frame(res)
