# -*- coding: utf-8 -*-
"""
@author: SURAJ
"""

import os
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import keras_tuner as kt
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from utils.SurvivalUtils import km_plot, h_vae_shap_values
from utils.DataLoader import load_data
from models.HVAECoxmodel import H_VAE_Cox_model
import matplotlib.pyplot as plt
import pickle
import random
from sksurv.metrics import concordance_index_ipcw
import shap

# tf.config.run_functions_eagerly(True) # used to show numpy values in Tensor object
earlystopper = EarlyStopping(monitor='val_cindex', mode='max', patience=20, verbose=1)
SEED = 20
random.seed(SEED)
feature_dataset, clinical_features, y_event, y_survival_time = load_data('lat_image_gene')
luad_feature_dataset, luad_clinical_features, luad_y_event, luad_y_survival_time = load_data('TCGA_LUAD_lat_image_gene')
lusc_feature_dataset, lusc_clinical_features, lusc_y_event, lusc_y_survival_time = load_data('TCGA_LUSC_lat_image_gene')

luad_feature_dataset = pd.DataFrame(StandardScaler().fit_transform(luad_feature_dataset),
                                    index=luad_feature_dataset.index, columns=luad_feature_dataset.columns)
lusc_feature_dataset = pd.DataFrame(StandardScaler().fit_transform(lusc_feature_dataset),
                                    index=lusc_feature_dataset.index, columns=lusc_feature_dataset.columns)
luad_clinical_features = pd.DataFrame(MinMaxScaler().fit_transform(luad_clinical_features),
                                      index=luad_clinical_features.index, columns=luad_clinical_features.columns)
lusc_clinical_features = pd.DataFrame(MinMaxScaler().fit_transform(lusc_clinical_features),
                                      index=lusc_clinical_features.index, columns=lusc_clinical_features.columns)


h_vae_cox_model = H_VAE_Cox_model(feature_dataset.shape[1], clinical_features.shape[1])


""" 
K-Fold cross validation to fit the model
"""
BATCH_SIZE = 26
SEARCH_EPOCHS = 20
EPOCHS = 100
outer_fold = 5
MAX_TRIALS = 3
EXECUTION_PER_TRIAL = 1


def plot_label_clusters(z_mean, risk):
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=risk, s=100)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.show()


# evaluate a model using k-fold cross-validation
def nested_cross_validate_model(radio_genomics_df, clinical_features, y_event, y_time):
    c_ipw, c_index, best_hyperparameters, shap_values, pred_risk_list, high_risk_features_list, high_risk_clinical_list, time, event = list(
    ), list(), list(), list(), list(), list(), list(), list(), list()

    # prepare cross validation
    k_outer_fold = StratifiedKFold(outer_fold, shuffle=True, random_state=SEED)
    # enumerate splits
    i = 1
    # outer fold
    for train_test_ix, validation_idx in k_outer_fold.split(radio_genomics_df, y_event):

        train_test_rad = radio_genomics_df.loc[train_test_ix].reset_index(drop=True)
        validation_rad = radio_genomics_df.loc[validation_idx].reset_index(drop=True)
        train_test_clic = clinical_features.loc[train_test_ix].reset_index(drop=True)
        validation_clic = clinical_features.loc[validation_idx].reset_index(drop=True)
        train_test_event = y_event.loc[train_test_ix].reset_index(drop=True)
        validation_test_event = y_event.loc[validation_idx].reset_index(drop=True)
        train_test_time = y_time.loc[train_test_ix].reset_index(drop=True)
        validation_time = y_time.loc[validation_idx].reset_index(drop=True)

        train_test_rad = pd.DataFrame(StandardScaler().fit_transform(train_test_rad))
        validation_rad = pd.DataFrame(StandardScaler().fit_transform(validation_rad))
        train_test_clic = pd.DataFrame(MinMaxScaler().fit_transform(train_test_clic))
        validation_clic = pd.DataFrame(MinMaxScaler().fit_transform(validation_clic))

        train_rad, test_rad, train_clic, test_clic, train_event, test_event, train_time, test_time = \
            train_test_split(train_test_rad, train_test_clic, train_test_event, train_test_time,  test_size=0.15,
                             stratify=train_test_event, shuffle=True, random_state=SEED)

        tuner = kt.BayesianOptimization(
            hypermodel=h_vae_cox_model.build_model,
            objective=kt.Objective("val_loss", direction="min"),
            max_trials=MAX_TRIALS,
            overwrite=True,
            executions_per_trial=EXECUTION_PER_TRIAL,
            directory="hyper_dir",
            project_name='H-VAE-Cox_run{}'.format(i),
        )

        tuner.search(
            [train_rad, train_clic], [train_event, train_time],
            validation_data=([test_rad, test_clic], [test_event, test_time]),
            epochs=SEARCH_EPOCHS,
            batch_size=BATCH_SIZE,
            # callbacks=[earlystopper]
        )

        best_hps = tuner.get_best_hyperparameters(1)[0]
        best_hyperparameters.append(best_hps.values)
        print(best_hps.values)

        #best_model = tuner.get_best_models()[0]
        c_model = tuner.hypermodel.build(best_hps)
        history = c_model.fit(
            [train_test_rad, train_test_clic], [train_test_event, train_test_time],
            validation_data=([validation_rad, validation_clic], [validation_test_event, validation_time]),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[earlystopper]
        )

        c_idx = c_model.evaluate([validation_rad, validation_clic], [validation_test_event, validation_time], verbose=0)
        luad_c_idx = c_model.evaluate([luad_feature_dataset, luad_clinical_features], [luad_y_event, luad_y_survival_time], verbose=0)
        lusc_c_idx = c_model.evaluate([lusc_feature_dataset, lusc_clinical_features], [lusc_y_event, lusc_y_survival_time], verbose=0)

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.legend(['train_loss', 'test_loss_fold_{}'.format(i)], loc='upper right')
        plt.show()
        plt.plot(history.history['cox_loss'])
        plt.plot(history.history['val_cox_loss'])
        plt.legend(['cox_loss', 'test_cox_loss, _fold_{}'.format(i)], loc='upper right')
        plt.show()
        plt.plot(history.history['cindex'])
        plt.plot(history.history['val_cindex'])
        plt.legend(['c_index', 'test_cindex_fold_{}'.format(i)], loc='upper right')
        plt.text(0.5, 0.4, 'val_cindex: {}'.format(c_idx[0]), fontsize=10)
        plt.show()

        print('validation c_index: ', c_idx[0])
        # perform high risk and low risk analysis here..
        z_mean, z_std, z, pred_risk = c_model.encoder.predict([validation_rad, validation_clic])
        luad_z_mean, luad_z_std, luad_z, luad_pred_risk = c_model.encoder.predict([luad_feature_dataset, luad_clinical_features])
        lusc_z_mean, lusc_z_std, lusc_z, lusc_pred_risk = c_model.encoder.predict([lusc_feature_dataset, lusc_clinical_features])

        et_train = np.array([(np.array(train_test_event)[i], np.array(train_test_time)[i])
                             for i in range(len(train_test_event))], dtype=[('e', bool), ('t', float)])

        et_test = np.array([(np.array(validation_test_event)[i], np.array(validation_time)[i])
                            for i in range(len(validation_test_event))], dtype=[('e', bool), ('t', float)])
        luad_et_test = np.array([(np.array(luad_y_event)[i], np.array(luad_y_survival_time)[i])
                                 for i in range(len(luad_y_event))], dtype=[('e', bool), ('t', float)])
        lusc_et_test = np.array([(np.array(lusc_y_event)[i], np.array(lusc_y_survival_time)[i])
                                 for i in range(len(lusc_y_event))], dtype=[('e', bool), ('t', float)])

        c_ipw.append({"Test set": concordance_index_ipcw(et_train, et_test, pred_risk.reshape(-1))[0],
                      "LUAD set": concordance_index_ipcw(et_train, luad_et_test, luad_pred_risk.reshape(-1))[0],
                      "LUSC set": concordance_index_ipcw(et_train, lusc_et_test, lusc_pred_risk.reshape(-1))[0]})

        median_risk = np.median(pred_risk)

        Risk = np.where(pred_risk > median_risk, 1, 0)
        plot_label_clusters(z_mean, Risk)

        validation_rad['Hazard'] = pred_risk
        high_risk_features = validation_rad[validation_rad['Hazard'] > median_risk]
        high_risk_clinical = validation_clic.loc[high_risk_features.index]
        high_risk_features.drop('Hazard', axis=1, inplace=True)
        shap_values.append(h_vae_shap_values(c_model, np.array(high_risk_features), np.array(high_risk_clinical)))

        pred_risk_list.append(pred_risk)
        high_risk_features_list.append(high_risk_features)
        high_risk_clinical_list.append(high_risk_clinical)

        print(high_risk_features.index)
        time.append(validation_time)
        event.append(validation_test_event)
        # c_index.append(c_idx[0])

        c_index.append({"Test_set": c_idx[0],
                        "LUAD_set": luad_c_idx[0],
                        "LUSC_set": lusc_c_idx[0]})
        i = i+1
    return c_ipw, c_index, best_hyperparameters, pred_risk_list, shap_values, time, event, high_risk_features_list, high_risk_clinical_list


c_ipw, c_index, best_hyperparameters, pred_risk_list, shap_values, time, event, high_risk_features_list, high_risk_clinical_list = nested_cross_validate_model(
    pd.DataFrame(np.array(feature_dataset)), pd.DataFrame(np.array(clinical_features)), y_event, y_survival_time)

# print('mean c_index', np.mean(c_index))
# print('std c_index', np.std(c_index))
print(pd.DataFrame(c_index))
print(pd.DataFrame(c_ipw))


# %%
# save trained objects

with open('SavedResults/H-VAE-CoxModel/hvae-coxmodelwithout_shap.pkl', 'wb') as res:
    pickle.dump((
        c_index, best_hyperparameters, pred_risk_list, shap_values, time, event, high_risk_features_list, high_risk_clinical_list), res)

# %%

hazard_list = np.array(pred_risk_list).reshape(-1)
time_list = np.array(time).reshape(-1)
event_list = np.array(event).reshape(-1)

km_plot(pd.DataFrame(event_list, columns=['Survival Status']), pd.DataFrame(time_list, columns=['survival_time']), hazard_list)

median_risk = np.median(hazard_list)
df = pd.DataFrame()
df['Hazard'] = np.array(hazard_list)
df['Risk'] = np.where(df.Hazard > median_risk, 1, 0)
risk_group = pd.concat([pd.DataFrame(event_list, columns=['Survival Status']), pd.DataFrame(
    time_list, columns=['survival_time']), df['Hazard'], df['Risk']], axis=1)
risk_group.to_csv('Results/H_VAE_Cox/H-VAE-Survival-Risk-Group.csv')


# %%

"""
SHAP interpretation

"""


class Explanation:
    def __init__(self, value, features):
        self.values = np.array(value)
        self.data = np.array(features)
        self.shape = np.array(value).shape
        self.base_values = None  # np.array(value)
        self.feature_names = features.columns


# append list of Gene Shap values
shap_obj = shap_values
gene_sha = pd.DataFrame()
for i in range(5):
    gene_sha = pd.concat([gene_sha, (pd.DataFrame(shap_obj[i][0][0][0]))])

# append high risk RNA smaples
rna_obj = high_risk_features_list
rna = []
rna = pd.DataFrame(rna_obj[0])
for i in range(1, 5):
    rna = pd.concat([rna, (pd.DataFrame(rna_obj[i]))])

rna.columns = feature_dataset.columns
gene_sha.columns = feature_dataset.columns

# shap for gene expression
shap_val_Exp = Explanation(gene_sha, rna)
shap.summary_plot(shap_val_Exp, max_display=40, show=False)
plt.rcParams['font.size'] = '20'
plt.savefig('Results/H_VAE_Cox/H-vae-shap_lat_feat.pdf')


# %%

"""
#SHAP INTERPRETATION

"""

shap_obj = shap_values

# image gene shap values
img_gene_sha = pd.DataFrame()
for i in range(5):
    img_gene_sha = pd.concat([img_gene_sha, (pd.DataFrame(shap_obj[i][0][0][0]))])

# clinical shap values
clinical_shap = pd.DataFrame()
for i in range(5):
    clinical_shap = pd.concat([clinical_shap, (pd.DataFrame(shap_obj[i][0][0][1]))])


img_rna_obj = pd.DataFrame()
for i in range(5):
    img_rna_obj = pd.concat([img_rna_obj, pd.DataFrame(high_risk_features_list[i])])


clinical_obj = pd.DataFrame()
for i in range(5):
    clinical_obj = pd.concat([clinical_obj, pd.DataFrame(high_risk_clinical_list[i])])

shap_values_extracted = pd.DataFrame(img_gene_sha)

image_shap_values = shap_values_extracted[shap_values_extracted.columns[:500]]
gene_shap_values = shap_values_extracted[shap_values_extracted.columns[500:]]


pd.DataFrame(image_shap_values).to_csv('Results/H_VAE_Cox/H-VAE-image_shap_values.csv')
pd.DataFrame(gene_shap_values).to_csv('Results/H_VAE_Cox/H-VAE-gene_shap_values.csv')
pd.DataFrame(clinical_shap).to_csv('Results/H_VAE_Cox/H-VAE-clinical_shap_values.csv')

# radiogenomics
img_rna_latents = pd.DataFrame(img_rna_obj)
image_latent = img_rna_latents[img_rna_latents.columns[:500]]
gene_latent = img_rna_latents[img_rna_latents.columns[500:]]

pd.DataFrame(image_shap_values).to_csv('Results/H_VAE_Cox/hvae_high_risk_image_latent.csv')
pd.DataFrame(gene_shap_values).to_csv('Results/H_VAE_Cox/hvae_high_risk_gene_latent.csv')
pd.DataFrame(clinical_obj).to_csv('Results/H_VAE_Cox/hvae_high_risk_clinical.csv')
