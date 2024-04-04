# -*- coding: utf-8 -*-
"""

@author: suraj

"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle


def load_data(data_type):

    if data_type == 'X-TCGA-LUSC':

        # Load saved cropped ROI Images

        img = ''
        with open('SavedObjects/TCGA-LUSC_CT_images.pkl', 'rb') as f:
            img = pickle.load(f)
        img.shape

        # read RNA, and clinical dataset
        rnaseq_raw_df = pd.read_csv('Data/TCGA_LUSC_RNA_seq.csv', index_col=0, header=0)
        rnaseq_raw_df.sort_index(inplace=True)

        clinical_df = pd.read_csv('Data/all_cohert_clinical_data.csv', index_col=0)

        # get common clinical and RNA seq dataset
        common_samples = list(set(clinical_df.index) &
                              set(rnaseq_raw_df.index))
        print(len(common_samples))

        filtered_clinical_df = clinical_df.loc[common_samples, :]
        filtered_clinical_df.sort_index(inplace=True)

        clinical_features = filtered_clinical_df[[
            'Age at Histological Diagnosis', 'Gender', 'Pathological T stage', 'Pathological N stage', 'Pathological M stage', 'Recurrence']]

        y_event = filtered_clinical_df[[
            'Survival Status']].reset_index(drop=True)
        y_survival_time = filtered_clinical_df[[
            'survival_time']].reset_index(drop=True)

        clic_columns = clinical_features.columns
        clic_index = clinical_features.index

        pathway_mask = pd.read_csv('Data/pathway_mask.csv', index_col=0)
        pathway_genes = pathway_mask.index
        rnaseq_scaled_df = rnaseq_raw_df.filter(pathway_genes)
        pathway_mask = pathway_mask.filter(rnaseq_raw_df.columns, axis=0)

        return img, rnaseq_scaled_df, pathway_mask, clinical_features, y_event, y_survival_time

    if data_type == 'X-TCGA-LUAD':

        # Load saved cropped ROI Images

        img = ''
        with open('SavedObjects/TCGA-LUAD_CT_images.pkl', 'rb') as f:
            img = pickle.load(f)
        img.shape

        # read RNA, and clinical dataset
        rnaseq_raw_df = pd.read_csv('Data/TCGA_LUAD_RNA_seq.csv', index_col=0, header=0)
        rnaseq_raw_df.sort_index(inplace=True)

        clinical_df = pd.read_csv('Data/all_cohert_clinical_data.csv', index_col=0)

        # get common clinical and RNA seq dataset
        common_samples = list(set(clinical_df.index) &
                              set(rnaseq_raw_df.index))
        print(len(common_samples))

        filtered_clinical_df = clinical_df.loc[common_samples, :]
        filtered_clinical_df.sort_index(inplace=True)

        clinical_features = filtered_clinical_df[[
            'Age at Histological Diagnosis', 'Gender', 'Pathological T stage', 'Pathological N stage', 'Pathological M stage', 'Recurrence']]

        y_event = filtered_clinical_df[[
            'Survival Status']].reset_index(drop=True)
        y_survival_time = filtered_clinical_df[[
            'survival_time']].reset_index(drop=True)

        clic_columns = clinical_features.columns
        clic_index = clinical_features.index

        pathway_mask = pd.read_csv('Data/pathway_mask.csv', index_col=0)
        pathway_genes = pathway_mask.index
        rnaseq_scaled_df = rnaseq_raw_df.filter(pathway_genes)
        pathway_mask = pathway_mask.filter(rnaseq_raw_df.columns, axis=0)

        return img, rnaseq_scaled_df, pathway_mask, clinical_features, y_event, y_survival_time

    if data_type == 'X-img-gene-clic':

        # Load saved cropped ROI Images

        img_1 = ''
        with open('SavedObjects/chunked_cropped_images_part1.pkl', 'rb') as f:
            img_1 = pickle.load(f)
        img_1.shape

        img_2 = ''
        with open('SavedObjects/chunked_cropped_images_part2.pkl', 'rb') as f:
            img_2 = pickle.load(f)
        img_2.shape

        img_3 = ''
        with open('SavedObjects/chunked_cropped_images_part3.pkl', 'rb') as f:
            img_3 = pickle.load(f)
        img_3.shape

        img_4 = ''
        with open('SavedObjects/chunked_cropped_images_part4.pkl', 'rb') as f:
            img_4 = pickle.load(f)
        img_4.shape

        img = np.concatenate((img_1, img_2, img_3, img_4))

        # read RNA, and clinical dataset
        rnaseq_raw_df = pd.read_csv('Data/sorted_RNA_Seq.csv', index_col=0, header=0)
        rnaseq_raw_df.sort_index(inplace=True)

        clinical_df = pd.read_csv('Data/all_cohert_clinical_data.csv', index_col=0)

        # get common clinical and RNA seq dataset
        common_samples = list(set(clinical_df.index) &
                              set(rnaseq_raw_df.index))
        print(len(common_samples))

        filtered_clinical_df = clinical_df.loc[common_samples, :]
        filtered_clinical_df.sort_index(inplace=True)

        clinical_features = filtered_clinical_df[[
            'Age at Histological Diagnosis', 'Gender', 'Pathological T stage', 'Pathological N stage', 'Pathological M stage', 'Recurrence']]

        y_event = filtered_clinical_df[[
            'Survival Status']]  # .reset_index(drop=True)
        y_survival_time = filtered_clinical_df[[
            'survival_time']]  # .reset_index(drop=True)

        clic_columns = clinical_features.columns
        clic_index = clinical_features.index

        pathway_mask = pd.read_csv('Data/pathway_mask.csv', index_col=0)
        pathway_genes = pathway_mask.index
        rnaseq_scaled_df = rnaseq_raw_df.filter(pathway_genes)
        pathway_mask = pathway_mask.filter(rnaseq_raw_df.columns, axis=0)

        return img, rnaseq_scaled_df, pathway_mask, clinical_features, y_event, y_survival_time

    elif data_type == 'lat_image_gene':

        rnaseq_scaled_df = pd.read_csv(
            'Results/cox_gene_latent.csv', index_col=0, header=0)
        image_scaled_df = pd.read_csv(
            'Results/cox_image_latent.csv', index_col=0, header=0)
        clinical_df = pd.read_csv('Data/all_cohert_clinical_data.csv', index_col=0)

        feature_dataset = np.concatenate(
            (image_scaled_df, rnaseq_scaled_df), axis=1)

        columns = []
        for col in range(image_scaled_df.shape[1]):
            columns.append('Image_LF_' + str(col))
        for col in range(rnaseq_scaled_df.shape[1]):
            columns.append('Gene_LF_' + str(col))
        feature_dataset = pd.DataFrame(
            feature_dataset, columns=columns, index=rnaseq_scaled_df.index)
        # get common clinical and RNA seq dataset
        common_samples = list(set(clinical_df.index) &
                              set(rnaseq_scaled_df.index))
        print(len(common_samples))

        filtered_clinical_df = clinical_df.loc[common_samples, :]
        filtered_clinical_df.sort_index(inplace=True)

        clinical_features = filtered_clinical_df[[
            'Age at Histological Diagnosis', 'Gender', 'Pathological T stage', 'Pathological N stage', 'Pathological M stage', 'Recurrence']]

        # clinical_features.to_csv('Data/filtered_clinical_data.csv')
        y_event = filtered_clinical_df[[
            'Survival Status']].reset_index(drop=True)
        y_survival_time = filtered_clinical_df[[
            'survival_time']].reset_index(drop=True)

        clic_columns = clinical_features.columns
        clic_index = clinical_features.index

        return feature_dataset, clinical_features, y_event, y_survival_time

    elif data_type == 'TCGA_LUAD_lat_image_gene':

        rnaseq_scaled_df = pd.read_csv(
            'Results/tcga_luad_cox_gene_latent.csv', index_col=0, header=0)
        image_scaled_df = pd.read_csv(
            'Results/tcga_luad_cox_image_latent.csv', index_col=0, header=0)
        clinical_df = pd.read_csv('Data/all_cohert_clinical_data.csv', index_col=0)

        feature_dataset = np.concatenate(
            (image_scaled_df, rnaseq_scaled_df), axis=1)
        # feature_dataset = StandardScaler().fit_transform(feature_dataset)
        columns = []
        for col in range(image_scaled_df.shape[1]):
            columns.append('LUAD_Image_LF_' + str(col))
        for col in range(rnaseq_scaled_df.shape[1]):
            columns.append('LUAD_Gene_LF_' + str(col))
        feature_dataset = pd.DataFrame(
            feature_dataset, columns=columns, index=rnaseq_scaled_df.index)
        # get common clinical and RNA seq dataset
        common_samples = list(set(clinical_df.index) &
                              set(rnaseq_scaled_df.index))
        print(len(common_samples))

        filtered_clinical_df = clinical_df.loc[common_samples, :]
        filtered_clinical_df.sort_index(inplace=True)

        clinical_features = filtered_clinical_df[[
            'Age at Histological Diagnosis', 'Gender', 'Pathological T stage', 'Pathological N stage', 'Pathological M stage', 'Recurrence']]

        y_event = filtered_clinical_df[[
            'Survival Status']].reset_index(drop=True)
        y_survival_time = filtered_clinical_df[[
            'survival_time']].reset_index(drop=True)

        clic_columns = clinical_features.columns
        clic_index = clinical_features.index

        return feature_dataset, clinical_features, y_event, y_survival_time

    elif data_type == 'TCGA_LUSC_lat_image_gene':

        # rnaseq_scaled_df = pd.read_csv('Results/downsampled_rna_pathway_simple.csv', index_col=0, header=0)
        rnaseq_scaled_df = pd.read_csv(
            'Results/tcga_lusc_cox_gene_latent.csv', index_col=0, header=0)
        image_scaled_df = pd.read_csv(
            'Results/tcga_lusc_cox_image_latent.csv', index_col=0, header=0)
        clinical_df = pd.read_csv('Data/all_cohert_clinical_data.csv', index_col=0)

        feature_dataset = np.concatenate(
            (image_scaled_df, rnaseq_scaled_df), axis=1)
        # feature_dataset = StandardScaler().fit_transform(feature_dataset)
        columns = []
        for col in range(image_scaled_df.shape[1]):
            columns.append('LUSC_Image_LF_' + str(col))
        for col in range(rnaseq_scaled_df.shape[1]):
            columns.append('LUSC_Gene_LF_' + str(col))
        feature_dataset = pd.DataFrame(
            feature_dataset, columns=columns, index=rnaseq_scaled_df.index)
        # get common clinical and RNA seq dataset
        common_samples = list(set(clinical_df.index) &
                              set(rnaseq_scaled_df.index))
        print(len(common_samples))

        filtered_clinical_df = clinical_df.loc[common_samples, :]
        filtered_clinical_df.sort_index(inplace=True)

        clinical_features = filtered_clinical_df[[
            'Age at Histological Diagnosis', 'Gender', 'Pathological T stage', 'Pathological N stage', 'Pathological M stage', 'Recurrence']]

        # clinical_features.to_csv('Data/filtered_clinical_data.csv')
        y_event = filtered_clinical_df[[
            'Survival Status']].reset_index(drop=True)
        y_survival_time = filtered_clinical_df[[
            'survival_time']].reset_index(drop=True)

        clic_columns = clinical_features.columns
        clic_index = clinical_features.index

        return feature_dataset, clinical_features, y_event, y_survival_time

    elif data_type == 'gene':

        rnaseq_raw_df = pd.read_csv(
            'Data/sorted_RNA_Seq.csv', index_col=0, header=0)
        rnaseq_raw_df.sort_index(inplace=True)
        rnaseq_scaled_df = MinMaxScaler().fit_transform(rnaseq_raw_df)
        rnaseq_scaled_df = pd.DataFrame(
            rnaseq_scaled_df, columns=rnaseq_raw_df.columns)

        pathway_mask = pd.read_csv('Data/pathway_mask.csv', index_col=0)
        pathway_genes = pathway_mask.index
        rnaseq_scaled_df = rnaseq_scaled_df.filter(pathway_genes)
        pathway_mask = pathway_mask.filter(rnaseq_scaled_df.columns, axis=0)

        return rnaseq_scaled_df, pathway_mask

    elif data_type == 'images':
        """
        Load saved cropped ROI Images

        """
        img_1 = ''
        with open('SavedObjects/chunked_cropped_images_part1.pkl', 'rb') as f:
            img_1 = pickle.load(f)
        img_1.shape

        img_2 = ''
        with open('SavedObjects/chunked_cropped_images_part2.pkl', 'rb') as f:
            img_2 = pickle.load(f)
        img_2.shape

        img_3 = ''
        with open('SavedObjects/chunked_cropped_images_part3.pkl', 'rb') as f:
            img_3 = pickle.load(f)
        img_3.shape

        img_4 = ''
        with open('SavedObjects/chunked_cropped_images_part4.pkl', 'rb') as f:
            img_4 = pickle.load(f)
        img_4.shape

        img = np.concatenate((img_1, img_2, img_3, img_4))
        rnaseq_scaled_df = pd.read_csv(
            'Results/downsampled_rna_pathway_sample.csv', index_col=0, header=0)

        return img, rnaseq_scaled_df

    elif data_type == 'gene_clinical':
        rnaseq_raw_df = pd.read_csv(
            'Data/sorted_RNA_Seq.csv', index_col=0, header=0)
        rnaseq_raw_df.sort_index(inplace=True)

        clinical_df = pd.read_csv('Data/all_cohert_clinical_data.csv', index_col=0)

        # get common clinical and RNA seq dataset
        common_samples = list(set(clinical_df.index) &
                              set(rnaseq_raw_df.index))
        print(len(common_samples))

        filtered_clinical_df = clinical_df.loc[common_samples, :]
        filtered_clinical_df.sort_index(inplace=True)

        clinical_features = filtered_clinical_df[[
            'Age at Histological Diagnosis', 'Gender', 'Pathological T stage', 'Pathological N stage', 'Pathological M stage', 'Recurrence']]

        # clinical_features.to_csv('Data/filtered_clinical_data.csv')
        y_event = filtered_clinical_df[[
            'Survival Status']].reset_index(drop=True)
        y_survival_time = filtered_clinical_df[[
            'survival_time']].reset_index(drop=True)

        return rnaseq_raw_df, clinical_features, y_event, y_survival_time

    elif data_type == 'image_clinical':
        """
        Load saved cropped ROI Images

        """
        img_1 = ''
        with open('SavedObjects/chunked_cropped_images_part1.pkl', 'rb') as f:
            img_1 = pickle.load(f)
        img_1.shape

        img_2 = ''
        with open('SavedObjects/chunked_cropped_images_part2.pkl', 'rb') as f:
            img_2 = pickle.load(f)
        img_2.shape

        img_3 = ''
        with open('SavedObjects/chunked_cropped_images_part3.pkl', 'rb') as f:
            img_3 = pickle.load(f)
        img_3.shape

        img_4 = ''
        with open('SavedObjects/chunked_cropped_images_part4.pkl', 'rb') as f:
            img_4 = pickle.load(f)
        img_4.shape

        feature_dataset = np.concatenate((img_1, img_2, img_3, img_4))

        # read downsampled RNA and clinical dataset
        rnaseq_df = pd.read_csv(
            'Data/sorted_RNA_Seq.csv', index_col=0, header=0)
        clinical_df = pd.read_csv('Data/all_cohert_clinical_data.csv', index_col=0)

        common_samples = list(set(clinical_df.index) & set(rnaseq_df.index))
        print(len(common_samples))

        filtered_clinical_df = clinical_df.loc[common_samples, :]
        filtered_clinical_df.sort_index(inplace=True)
        # pd.DataFrame(filtered_clinical_df).to_csv('Data/sorted_clinic_index.csv')

        # labels = filtered_clinical_df['Histology ']
        # pd.DataFrame(labels).to_csv('Data/labels.csv')

        filtered_clinical_df = clinical_df.loc[common_samples, :]
        filtered_clinical_df.sort_index(inplace=True)

        clinical_features = filtered_clinical_df[[
            'Age at Histological Diagnosis', 'Gender', 'Pathological T stage', 'Pathological N stage', 'Pathological M stage', 'Recurrence']]

        # clinical_features.to_csv('Data/filtered_clinical_data.csv')
        # .reset_index(drop=True)
        y_event = filtered_clinical_df[['Survival Status']]
        y_survival_time = filtered_clinical_df[[
            'survival_time']].reset_index(drop=True)

        return feature_dataset, clinical_features, y_event, y_survival_time

    elif data_type == 'TCGA_LUAD_image_clinical':
        """
        Load saved cropped ROI Images

        """
        feature_dataset = ''
        with open('SavedObjects/TCGA-LUAD_CT_images.pkl', 'rb') as f:
            feature_dataset = pickle.load(f)
        feature_dataset.shape

        # read downsampled RNA and clinical dataset
        rnaseq_df = pd.read_csv(
            'Data/TCGA_LUAD_RNA_seq.csv', index_col=0, header=0)
        clinical_df = pd.read_csv('Data/all_cohert_clinical_data.csv', index_col=0)

        common_samples = list(set(clinical_df.index) & set(rnaseq_df.index))
        print(len(common_samples))

        filtered_clinical_df = clinical_df.loc[common_samples, :]
        filtered_clinical_df.sort_index(inplace=True)

        filtered_clinical_df = clinical_df.loc[common_samples, :]
        filtered_clinical_df.sort_index(inplace=True)

        clinical_features = filtered_clinical_df[[
            'Age at Histological Diagnosis', 'Gender', 'Pathological T stage', 'Pathological N stage', 'Pathological M stage', 'Recurrence']]

        # clinical_features.to_csv('Data/filtered_clinical_data.csv')
        # .reset_index(drop=True)
        y_event = filtered_clinical_df[['Survival Status']]
        y_survival_time = filtered_clinical_df[[
            'survival_time']].reset_index(drop=True)

        return feature_dataset, clinical_features, y_event, y_survival_time

    elif data_type == 'TCGA_LUSC_image_clinical':
        """
        Load saved cropped ROI Images

        """
        feature_dataset = ''
        with open('SavedObjects/TCGA-LUSC_CT_images.pkl', 'rb') as f:
            feature_dataset = pickle.load(f)
        feature_dataset.shape

        # read downsampled RNA and clinical dataset
        rnaseq_df = pd.read_csv(
            'Data/TCGA_LUSC_RNA_seq.csv', index_col=0, header=0)
        clinical_df = pd.read_csv('Data/all_cohert_clinical_data.csv', index_col=0)

        common_samples = list(set(clinical_df.index) & set(rnaseq_df.index))
        print(len(common_samples))

        filtered_clinical_df = clinical_df.loc[common_samples, :]
        filtered_clinical_df.sort_index(inplace=True)

        filtered_clinical_df = clinical_df.loc[common_samples, :]
        filtered_clinical_df.sort_index(inplace=True)

        clinical_features = filtered_clinical_df[[
            'Age at Histological Diagnosis', 'Gender', 'Pathological T stage', 'Pathological N stage', 'Pathological M stage', 'Recurrence']]

        # clinical_features.to_csv('Data/filtered_clinical_data.csv')
        # .reset_index(drop=True)
        y_event = filtered_clinical_df[['Survival Status']]
        y_survival_time = filtered_clinical_df[[
            'survival_time']].reset_index(drop=True)

        return feature_dataset, clinical_features, y_event, y_survival_time

    elif data_type == 'gene-pathwaymask':

        rnaseq_raw_df = pd.read_csv('Data/sorted_RNA_Seq.csv', index_col=0, header=0)

        rnaseq_raw_df.sort_index(inplace=True)

        pathway_mask = pd.read_csv('Data/pathway_mask.csv', index_col=0)
        pathway_genes = pathway_mask.index
        rnaseq_scaled_df = rnaseq_raw_df.filter(pathway_genes)
        pathway_mask = pathway_mask.filter(rnaseq_scaled_df.columns, axis=0)

        clinical_df = pd.read_csv('Data/all_cohert_clinical_data.csv', index_col=0)

        # get common clinical and RNA seq dataset
        common_samples = list(set(clinical_df.index) &
                              set(rnaseq_scaled_df.index))
        print(len(common_samples))

        filtered_clinical_df = clinical_df.loc[common_samples, :]
        filtered_clinical_df.sort_index(inplace=True)

        y_event = filtered_clinical_df[[
            'Survival Status']].reset_index(drop=True)
        y_survival_time = filtered_clinical_df[[
            'survival_time']].reset_index(drop=True)

        return rnaseq_scaled_df, pathway_mask, y_event, y_survival_time

    elif data_type == 'lusc_gene-pathwaymask':
        rnaseq_raw_df = pd.read_csv('Data/TCGA_LUSC_RNA_seq.csv', index_col=0, header=0)
        rnaseq_raw_df.sort_index(inplace=True)

        pathway_mask = pd.read_csv('Data/pathway_mask.csv', index_col=0)
        pathway_genes = pathway_mask.index
        rnaseq_scaled_df = rnaseq_raw_df.filter(pathway_genes)
        pathway_mask = pathway_mask.filter(rnaseq_scaled_df.columns, axis=0)

        clinical_df = pd.read_csv('Data/all_cohert_clinical_data.csv', index_col=0)

        # get common clinical and RNA seq dataset
        common_samples = list(set(clinical_df.index) &
                              set(rnaseq_scaled_df.index))
        print(len(common_samples))

        filtered_clinical_df = clinical_df.loc[common_samples, :]
        filtered_clinical_df.sort_index(inplace=True)

        y_event = filtered_clinical_df[[
            'Survival Status']].reset_index(drop=True)
        y_survival_time = filtered_clinical_df[[
            'survival_time']].reset_index(drop=True)

        return rnaseq_scaled_df, pathway_mask, y_event, y_survival_time

    elif data_type == 'luad_gene-pathwaymask':
        rnaseq_raw_df = pd.read_csv('Data/TCGA_LUAD_RNA_seq.csv', index_col=0, header=0)
        rnaseq_raw_df.sort_index(inplace=True)

        pathway_mask = pd.read_csv('Data/pathway_mask.csv', index_col=0)
        pathway_genes = pathway_mask.index
        rnaseq_scaled_df = rnaseq_raw_df.filter(pathway_genes)
        pathway_mask = pathway_mask.filter(rnaseq_scaled_df.columns, axis=0)

        clinical_df = pd.read_csv('Data/all_cohert_clinical_data.csv', index_col=0)

        # get common clinical and RNA seq dataset
        common_samples = list(set(clinical_df.index) &
                              set(rnaseq_scaled_df.index))
        print(len(common_samples))

        filtered_clinical_df = clinical_df.loc[common_samples, :]
        filtered_clinical_df.sort_index(inplace=True)

        y_event = filtered_clinical_df[[
            'Survival Status']].reset_index(drop=True)
        y_survival_time = filtered_clinical_df[[
            'survival_time']].reset_index(drop=True)

        return rnaseq_scaled_df, pathway_mask, y_event, y_survival_time

    elif data_type == 'image-autoencoder':
        """
        Load saved cropped ROI Images

        """
        img_1 = ''
        with open('SavedObjects/chunked_cropped_images_part1.pkl', 'rb') as f:
            img_1 = pickle.load(f)
        img_1.shape

        img_2 = ''
        with open('SavedObjects/chunked_cropped_images_part2.pkl', 'rb') as f:
            img_2 = pickle.load(f)
        img_2.shape

        img_3 = ''
        with open('SavedObjects/chunked_cropped_images_part3.pkl', 'rb') as f:
            img_3 = pickle.load(f)
        img_3.shape

        img_4 = ''
        with open('SavedObjects/chunked_cropped_images_part4.pkl', 'rb') as f:
            img_4 = pickle.load(f)
        img_4.shape

        img = np.concatenate((img_1, img_2, img_3, img_4))
        rna_latent_features = pd.read_csv(
            'Results/cox_gene_latent.csv', index_col=0, header=0)

        return img, rna_latent_features
