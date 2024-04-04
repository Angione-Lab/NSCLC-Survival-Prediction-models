install.packages('survival') # install package
library(survminer)
library(survival)
ValidationTest <- function(PI, groupRisk, time, status){
  
  PI$groupRisk <- groupRisk
  PI$time <- time
  PI$status <- status
  #browser()
  
  if((length(unique(groupRisk)) > 1)==TRUE){

    logranktest <- survdiff(Surv(time, status) ~ factor(groupRisk), data = PI, rho = 0)
    p.quantile <- 1-pchisq(logranktest$chisq, 1)
    p.value <- format(p.quantile, scientific = TRUE) # signif(p.quantile,3)
    
    # Kaplan-Meier survival curves
    fitTest <- survfit(Surv(time, status) ~ factor(groupRisk), data = PI)
    library(ggplot2)
    survp <- survminer::ggsurvplot(
      fitTest,                  
      data = PI,  
      conf.int = TRUE,           
      pval = p.value,               
      risk.table = TRUE,
      #surv.median.line = "hv",
      ggtheme = theme_minimal(), 
     
      
      # legend.title = paste0("Optimal Cutoff: ", opt.cutoff),
      legend.labs = c("Low Risk","High Risk"))
    
    # print(survp)
     print(survp, newpage = TRUE)

    
  } else {p.value <- 1
  print("Warning: no splitting!")} # no splitting - only one group
  
# } 
  
  df <- PI
  return(list(df=df, p.value=p.value))
}

#risk_file = 'C:\\1.workingDirectory\\NSCLCSruvivalPrediction\\Results\\SurvivalResults\\GeneOnlySurvivalRiskGroup.csv'
#risk_file = 'C:\\1.workingDirectory\\NSCLCSruvivalPrediction\\Results\\SurvivalResults\\GeneClinical-SurvivalRiskGroup.csv'
#risk_file = 'C:\\1.workingDirectory\\NSCLCSruvivalPrediction\\Results\\SurvivalResults\\ImageOnlySurvivalRiskGroup.csv'
#risk_file = 'C:\\1.workingDirectory\\NSCLCSruvivalPrediction\\Results\\SurvivalResults\\ImageClinicalSurvivalRiskGroup.csv'
#risk_file = 'C:\\1.workingDirectory\\NSCLCSruvivalPrediction\\Results\\SurvivalResults\\HVAESurvivalRiskGroup.csv'
#risk_file = 'C:\\1.workingDirectory\\NSCLCSruvivalPrediction\\Results\\SurvivalResults\\XVAESurvivalRiskGroup.csv'
#risk_file = 'D:/NSCLC_try/updates_multiple_cohert/NSCLC-Survival-Prediction-models/Results/only_clinical-Risk-Group.csv'
risk_file = 'Results/Att_X-VAE-Survival-Risk-Group2.csv'
mydata <- read.table(risk_file, row.names=1,sep=',', header=TRUE)

ValidationTest(data.frame(mydata$Hazard), mydata$Risk, mydata$survival_time, mydata$Survival.Status)



