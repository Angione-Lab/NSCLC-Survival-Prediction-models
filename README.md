# NSCLC-Survival-Prediction-models

This repository contains the code and data to reproduce the results presented in the paper: S. Verma, G. Magazzù, N. Eftekhari, A. Occhipinti, C. Angione, "Small-sample deep learning on joint omics-imaging-clinical data of 130 non-small cell lung cancer patients".

H-VAE-Cox and XAT-VAE-Cox are pathway-guided biologically interpretable survival prediction models for Non-Small Cell Lung Cancer. The survival prediction is performed using radiological images (CT scan images), gene expression and clinical information. 

Radiogenomics (CT scan images and Gene Expression) data along with clinical data are collected from TCIA and GEO. The collected data is then preprocessed. To begin with, the region of interest (ROI) i.e. tumour region is segmented using U-Net, the null values from gene expression data are removed, and the resulting data is normalised. Feature selection is performed on images, gene expression data and clinical data, and common samples from all the datasets are selected to be fed into deep learning models. 

The deep learning models (H-VAE-Cox and X-VAE-Cox models) estimate the prognostic index (PI) using images, gene expression and clinical data. The results from the models are evaluated using C-index and Kaplan-Meier curves, and plotted for high and low-risk group patients. The models were interpreted using SHAP values. Finally, the significant genes identified by SHAP were biologically interpreted using KEGG and Reactome.


## Steps to run the H-VAE-Cox and X-VAE-Cox models for survival prediction ##
To get started, download the required data from the links provided below and save it in the appropriate folders, as detailed below.

*	The raw CT scan images and segments can be downloaded from https://figshare.com/articles/dataset/NSCLC_CT_Scan_Images/17946047 . Save the CT scan images and Segments in the "Data/NSCLC_Images" folder of the project as illustrated in Figure 1.
* The pre-processed ROI-extracted CT scan images can be downloaded from https://figshare.com/articles/dataset/ROI_extracted_from_CT_Scan_images/17940776. Save these files in the "SavedObjects" folder of the project as illustrated in Figure 1.
*	Download the Gene Expression, Pathway mask and clinical data from https://figshare.com/articles/dataset/Data/17941775. Save these files in the "Data" folder of the project.

To pre-process the raw CT scan images and extract the ROI, execute  [run/U-net-ImageSegmentation.py script](https://github.com/Angione-Lab/NSCLC-Survival-Prediction-models/blob/main/run/U-net-ImageSegmentation.py)


Note: ImageSegmentation can be skipped if you use the pre-processed ROI extracted CT scan images.


## For H-VAE-Cox model survival prediction ##

*	Step 1: Run [run/GenerateGeneLatentFeatures.py] (https://github.com/Angione-Lab/NSCLC-Survival-Prediction-models/blob/main/run/GenerateGeneLatentFeatures.py) to generate lower-dimensional latent vectors for the gene expression dataset and perform survival prediction with the gene expression data only.
*	Step 2: Run [run/GenerateGeneImageFeatures.py] (https://github.com/Angione-Lab/NSCLC-Survival-Prediction-models/blob/main/run/GenerateImageLatentFeatures.py) to generate lower-dimensional latent vectors for the preprocessed ROI extracted from CT scan images and perform survival prediction with the imaging data only.
*	Step 3: Run [run/RunHvaeCoxModel.py] (https://github.com/Angione-Lab/NSCLC-Survival-Prediction-models/blob/main/run/RunHvaeCoxModel.py) for survival prediction using the latent features generated from step 1, 2, and clinical data.
*	Step 4: Run [run/H_VAE_Shap_Interpretation.py] (https://github.com/Angione-Lab/NSCLC-Survival-Prediction-models/blob/main/run/H_VAE_Shap_Interpretation.py) script for the SHAP interpretation of the H-VAE-Cox model. 

Note: While performing SHAP interpretation in Step 3, please make sure the gene expression autoencoder and the image autoencoder from Step 1 are saved as .h5 so that the saved models can be reused.

## For XAT-VAE-Cox model survival prediction using CT Scan images, gene expression and clinical information ##
Step 1: Run [run/RunXVaeCoxModel.py](https://github.com/Angione-Lab/NSCLC-Survival-Prediction-models/blob/main/run/RunXVaeCoxModel.py) for survival prediction and SHAP interpretation.

## For XAT-VAE-Cox model survival prediction using CT scan images and clinical information ##
Step: Run [run/RunImageClinicalCoxModel.py] (https://github.com/Angione-Lab/NSCLC-Survival-Prediction-models/blob/main/run/RunImageClinicalCoxModel.py)

## For XAT-VAE-Cox model survival prediction using Gene Expression and clinical information ##
Step: [run/RunGeneClinicalCoxModel.py] (https://github.com/Angione-Lab/NSCLC-Survival-Prediction-models/blob/main/run/RunGeneClinicalCoxModel.py)

## Plots ##
Kaplan Meier plots and biological pathway interpretation plots are plotted in R.

## Project Structure ##
![Project Structure](https://github.com/Angione-Lab/NSCLC-Survival-Prediction-models/blob/main/Results/project%20structure.png)
