# -*- coding: utf-8 -*-
"""
@author: Suraj
"""

from models.UnetModel import UnetModel
from utils.ImageProcessingUtils import ImagePreprocessing
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.layers import Input
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


LOAD_SAVED_PICKLE = False
IMG_SIZE_PX = 224
SLICE_COUNT = 5

cropped_roi = []


def predict_segments(imgs):
    """
    Segment the tumour region and extract (Crop) the Region of Interest

    Returns
    -------
    TYPE
        DESCRIPTION. list of cropped tumour region from CT-scan slices.

    """
    for idx, x in enumerate(imgs):
        print(x.shape)
        pred = segment_model.predict(np.array(x))
        print(pred.shape)
        roi = imagePreprocessing.extract_ROI(x, pred)
        cropped_roi.append(roi)
    return np.array(cropped_roi)


# Load Radiogenomics dataset
rna_exp = pd.read_csv(r'Data/sorted_RNA_Seq.csv', index_col=0)

# Load TCGA dataset
luad_df = pd.read_csv(r'D:\NSCLC_datasets\TCGA_Lung_dataset\lusc_RNA_seq', sep='\t', index_col=0).T
luad_images = os.listdir(r'D:\NSCLC_datasets\TCGA_Lung_dataset\TCGA-LUSC')
luad_df['Patient_id'] = luad_df.index.str[:-3]
luad_df.drop_duplicates(subset=['Patient_id'], inplace=True)
common_samples_luad = set(luad_images).intersection(set(luad_df['Patient_id']))
luad_df = luad_df[luad_df['Patient_id'].isin(common_samples_luad)]
common_samples_luad = sorted(common_samples_luad, reverse=False)
luad_df.sort_index(inplace=True)
luad_df.index = luad_df['Patient_id']

common_genes = np.sort(list(set(rna_exp.columns).intersection(set(luad_df.columns))))
luad_df[common_genes].to_csv(r'Data/TCGA_LUSC_RNA_seq.csv')

rna_exp[common_genes].to_csv(r'Data/sorted_RNA_Seq.csv')


slice_path = 'D:/NSCLC_datasets/TCGA_Lung_dataset/TCGA-LUSC/'
patients = os.listdir(slice_path)
imagePreprocessing = ImagePreprocessing(IMG_SIZE_PX, IMG_SIZE_PX, SLICE_COUNT, slice_path)

# initialise U-Net model
umodel = UnetModel(IMG_SIZE_PX, IMG_SIZE_PX)
input_image = Input(shape=(IMG_SIZE_PX, IMG_SIZE_PX, 1))
segment_model = umodel.model(input_image)
segment_model.summary()
segment_model.load_weights('SavedObjects/u_net_model.h5')

imgs = imagePreprocessing.load_patientwise_slices(patients)

# %%
crp_roi = predict_segments(imgs)

print(crp_roi.shape)
plt.imshow(tf.squeeze(crp_roi[0][:, :, 1]), cmap='gray')


# %%
# save the cropped ROI as pickle object
with open('SavedObjects/TCGA-LUSC_CT_images.pkl', 'wb') as chunked_cropped_images:
    pickle.dump(crp_roi, chunked_cropped_images)

# %%


def prepare_tcga_data(cohert):
    # load tcga luad data
    if cohert == 'LUAD':
        tcga_rna = pd.read_csv(r'Data/TCGA_LUAD_RNA_seq.csv', index_col=0)
        tcga_clinical = pd.read_csv(r'D:\NSCLC_datasets\TCGA_Lung_dataset\luad_clinical_data.txt', sep='\t', index_col=1)
    else:
        tcga_rna = pd.read_csv(r'Data/TCGA_LUSC_RNA_seq.csv', index_col=0)
        tcga_clinical = pd.read_csv(r'D:\NSCLC_datasets\TCGA_Lung_dataset\lusc_clinical_data.txt', sep='\t', index_col=1)

    common_samples = set(tcga_rna.index).intersection(set(tcga_clinical.index))

    tcga_clinical = tcga_clinical[['Diagnosis Age',  'Sex', 'American Joint Committee on Cancer Tumor Stage Code',
                                   'Neoplasm Disease Lymph Node Stage American Joint Committee on Cancer Code',
                                   'American Joint Committee on Cancer Metastasis Stage Code', 'Disease Free Status',
                                   'Overall Survival Status', 'Overall Survival (Months)']]

    tcga_clinical = tcga_clinical[tcga_clinical.index.isin(common_samples)]
    tcga_clinical['Overall Survival (Months)'] = tcga_clinical['Overall Survival (Months)'].astype(float) * 30
    tcga_clinical['Overall Survival (Months)'] = tcga_clinical['Overall Survival (Months)'].astype(int)

    tcga_clinical['American Joint Committee on Cancer Metastasis Stage Code'].replace(
        ["[Not Available]"], tcga_clinical['American Joint Committee on Cancer Metastasis Stage Code'].value_counts().idxmax(), inplace=True)

    tcga_clinical['Disease Free Status'].replace(["0:DiseaseFree"], 0, inplace=True)
    tcga_clinical['Disease Free Status'].replace(["1:Recurred/Progressed"],  1, inplace=True)

    tcga_clinical['Overall Survival Status'].replace(["0:LIVING"], 0, inplace=True)
    tcga_clinical['Overall Survival Status'].replace(["1:DECEASED"], 1, inplace=True)

    # tcga_clinical['Patient Smoking History Category'].replace(
    #    ["[Not Available]"], tcga_clinical['Patient Smoking History Category'].value_counts().idxmax(), inplace=True)

    tcga_clinical['Disease Free Status'].replace(
        ["[Not Available]"], tcga_clinical['Disease Free Status'].value_counts().idxmax(), inplace=True)

    tcga_clinical.columns = ['Age at Histological Diagnosis', 'Gender', 'Pathological T stage',
                             'Pathological N stage', 'Pathological M stage', 'Recurrence',
                             'Survival Status', 'Time to Death (days)']

    return tcga_clinical


luad_clinical_data = prepare_tcga_data('LUAD')
lusc_clinical_data = prepare_tcga_data('LUSC')

nsclc_clinical = pd.read_csv(r'Data/NSCLC_Clinical_Data.csv', index_col=0)
nsclc_clinical['Time to Death (days)'].fillna(nsclc_clinical['survival_time'], inplace=True)

nsclc_clinical = nsclc_clinical[['Age at Histological Diagnosis', 'Gender', 'Pathological T stage',
                                 'Pathological N stage', 'Pathological M stage', 'Recurrence',
                                 'Survival Status', 'Time to Death (days)']]

nsclc_clinical['Recurrence'].replace(["no"], 0, inplace=True)
nsclc_clinical['Recurrence'].replace(["yes"],  1, inplace=True)

nsclc_clinical['Survival Status'].replace(["Alive"], 0, inplace=True)
nsclc_clinical['Survival Status'].replace(["Dead"],  1, inplace=True)

nsclc_rna = pd.read_csv(r'Data/sorted_RNA_Seq.csv', index_col=0)

merged_nsclc_rna_clic = pd.merge(nsclc_rna, nsclc_clinical, left_on=nsclc_rna.index, right_on=nsclc_clinical.index)
merged_nsclc_rna_clic.index = merged_nsclc_rna_clic['key_0']
merged_nsclc_rna_clic.drop(['key_0'], axis=1, inplace=True)

all_clinical_data = pd.concat([luad_clinical_data, lusc_clinical_data, merged_nsclc_rna_clic[nsclc_clinical.columns]])

encoded_features = all_clinical_data[['Gender', 'Pathological T stage',
                                      'Pathological N stage', 'Pathological M stage']].apply(LabelEncoder().fit_transform)

encoded_features[['Age at Histological Diagnosis', 'Recurrence', 'Survival Status', 'survival_time']
                 ] = all_clinical_data[['Age at Histological Diagnosis', 'Recurrence', 'Survival Status', 'Time to Death (days)']]
encoded_features.to_csv('Data/all_cohert_clinical_data.csv')

# %%
nsclc_rna = pd.read_csv('Data/sorted_RNA_Seq.csv', index_col=0)
lusc = pd.read_csv('Data/TCGA_LUSC_RNA_seq.csv', index_col=0)
luad = pd.read_csv('Data/TCGA_LUAD_RNA_seq.csv', index_col=0)

Annotation_nsclc = pd.DataFrame([1 for i in range(nsclc_rna.shape[0])], index=nsclc_rna.index)
Annotation_lusc = pd.DataFrame([2 for i in range(lusc.shape[0])], index=lusc.index)
Annotation_luad = pd.DataFrame([3 for i in range(luad.shape[0])], index=luad.index)

rna_merged = pd.concat([nsclc_rna, lusc, luad])
annotation_merged = pd.concat([Annotation_nsclc, Annotation_lusc, Annotation_luad])

rna_merged.to_csv('Data/merged_rna_seq.csv')
annotation_merged.to_csv('Data/annotation_merged.csv')


