# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 23:30:46 2022

@author: SURAJ
"""


import tensorflow.keras as keras
import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle

"""
Decode Images, gene expression, clinical data and Shap values
"""
#load saved decoder model

GeneDecoder = keras.models.load_model('SavedObjects/H_VAE/GeneSparseDecoder_5.h5')
ImageDecoder = keras.models.load_model('SavedObjects/H_VAE/Image Decoder_5.h5')
hvae_high_risk_rna_seq = pd.read_csv('Results/H_VAE_Cox/hvae_high_risk_gene_latent.csv', index_col=0, header=0)
hvae_high_risk_images = pd.read_csv('Results/H_VAE_Cox/hvae_high_risk_image_latent.csv', index_col=0, header=0)
hvae_high_risk_clinical = pd.read_csv('Results/H_VAE_Cox/hvae_high_risk_clinical.csv', index_col=0, header=0)

raw_rna_seq = pd.read_csv('Data/sorted_RNA_Seq.csv', index_col=0)
pathway_mask = pd.read_csv('Data/pathway_mask.csv', index_col = 0)

gene_shap_values = pd.read_csv('Results/H_VAE_Cox/H-VAE-gene_shap_values.csv', index_col=0)
clinical_shap_values = pd.read_csv('Results/H_VAE_Cox/H-VAE-clinical_shap_values.csv',index_col = 0)

shap_geneDecoder_model = keras.Model(inputs= GeneDecoder.input, outputs = GeneDecoder.get_layer('gene_decoder').output)
constructed_gene_expression_shap = shap_geneDecoder_model.predict(gene_shap_values)
constructed_gene_expression_data = GeneDecoder.predict(hvae_high_risk_rna_seq)
constructed_gene_expression_shap = pd.DataFrame(constructed_gene_expression_shap, columns= raw_rna_seq.columns)
constructed_gene_expression_data = pd.DataFrame(constructed_gene_expression_data, columns= raw_rna_seq.columns)

#%%
class Explanation:
    def __init__(self, value, features):
        self.values = np.array(value)
        self.data = np.array(features)
        self.shape = np.array(value).shape
        self.base_values = None #np.array(value)
        self.feature_names = features.columns
         
shap_val = Explanation(constructed_gene_expression_shap,constructed_gene_expression_data)

shap.summary_plot(shap_val, max_display=40, show=False)
plt.tight_layout()
plt.savefig('Results/H_VAE_Cox/H-vae-shap-plot_2.pdf')

vals= np.abs(constructed_gene_expression_shap).mean(0)

feature_importance = pd.DataFrame(vals, columns=['Importance'])
feature_importance.sort_values(by=['Importance'], ascending=False,inplace=True)
feature_importance.to_csv('Results/H_VAE_Cox/H-VAE-Shap-Important-Genes_2.csv' )
feature_importance.head()

#%%
"""
SHAP interpretation for clinical data
"""
shap_val = Explanation(clinical_shap_values,hvae_high_risk_clinical)
shap.summary_plot(shap_val, max_display=50, show=False)
plt.savefig('Results/H_VAE_Cox/H-vae-shap-clinical_2.pdf')



clinical_shap_values.columns = hvae_high_risk_clinical.columns

vals= np.abs(clinical_shap_values).mean(0)

feature_importance = pd.DataFrame(vals, columns=['Importance'])
feature_importance.sort_values(by=['Importance'], ascending=False,inplace=True)
feature_importance.to_csv('Results/H_VAE_Cox/H-VAE-Shap-Important-Clinical_2.csv')
feature_importance.head()


#%%
"""
SHAP interpretation for images
"""
image_shap_values = pd.read_csv('Results/H_VAE_Cox/H-VAE-image_shap_values.csv', index_col=0)
dec_img_shap = ImageDecoder.predict(image_shap_values)

decode_images = ImageDecoder.predict(hvae_high_risk_images)
shap.image_plot([dec_img_shap[i][:,:,2] for i in range(10)], [decode_images[i][:,:,2] for i in range(10)],show=False)
#plt.savefig('Results/H_VAE_Cox/H-vae-img-shap.pdf')

#%%

shap_img = np.array([np.expand_dims(dec_img_shap[i][:,:,3] , axis=-1) for i in range(13)])
test_img = np.array([np.expand_dims(decode_images[i][:,:,3] , axis=-1) for i in range(13)])
shap.image_plot(shap_img, test_img,show=False)