# -*- coding: utf-8 -*-
"""
@author: SURAJ
"""
import numpy as np
import keras_tuner as kt
import pandas as pd
from sklearn.model_selection import KFold
import time
from utils.SurvivalUtils import km_plot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from utils.DataLoader import load_data
from models.GeneClinicalCoxmodel import X_VAE_GeneCox_model
import pickle
import random

random.seed(20)
batch_size = 26
search_epochs = 20
epochs = 100
max_trial = 3
exec_per_trial = 1
feature_dataset, clinical_features, y_event, y_survival_time = load_data('gene_clinical')


latent_dim = 256
original_dim = feature_dataset.shape[1]
clinical_dim = clinical_features.shape[1]


x_vae_genecox_model = X_VAE_GeneCox_model(original_dim, clinical_dim, latent_dim)

# evaluate a model using k-fold cross-validation


def nested_cross_validate_model(feature_df, clinical_features, y_event, y_time, n_folds=5):
    c_index, best_hyperparameters, shap_values, pred_risk_list, time, event, tuner_summary = list(), list(), list(), list(), list(), list(), list()

    # prepare cross validation
    k_outer_fold = KFold(n_folds, shuffle=True, random_state=5)
    # enumerate splits
    i = 1
    # outer fold
    for train_test_ix, validation_idx in k_outer_fold.split(feature_df):

        train_test_rna = feature_df.loc[train_test_ix].reset_index(drop=True)
        validation_rna = feature_df.loc[validation_idx].reset_index(drop=True)
        train_test_clic = clinical_features.loc[train_test_ix].reset_index(drop=True)
        validation_clic = clinical_features.loc[validation_idx].reset_index(drop=True)
        train_test_event = y_event.loc[train_test_ix].reset_index(drop=True)
        validation_test_event = y_event.loc[validation_idx].reset_index(drop=True)
        train_test_time = y_time.loc[train_test_ix].reset_index(drop=True)
        validation_time = y_time.loc[validation_idx].reset_index(drop=True)

        train_test_rna = MinMaxScaler().fit_transform(train_test_rna)
        validation_rna = MinMaxScaler().fit_transform(validation_rna)
        train_test_clic = MinMaxScaler().fit_transform(train_test_clic)
        validation_clic = MinMaxScaler().fit_transform(validation_clic)

        tuner = kt.BayesianOptimization(
            hypermodel=x_vae_genecox_model.build_model,
            objective=kt.Objective("val_cindex_metric", direction="max"),
            max_trials=max_trial,
            overwrite=True,
            executions_per_trial=exec_per_trial,
            directory="G_XVAE-hyper_dir",
            project_name='Gene_XVAE-Cox{}'.format(i),
        )
        # stratified train test split
        train_rna, test_rna, train_clic, test_clic, train_event, test_event, train_time, test_time = \
            train_test_split(train_test_rna, train_test_clic, train_test_event, train_test_time,  test_size=0.15,
                             stratify=train_test_event, shuffle=True, random_state=5)

        tuner.search([train_rna, train_clic], [train_event, train_time],
                     batch_size=batch_size,
                     epochs=search_epochs,
                     validation_data=([test_rna, test_clic], [test_event, test_time]))

        best_hps = tuner.get_best_hyperparameters(1)[0]
        best_hyperparameters.append(best_hps.values)
        print(best_hps.values)
        print(tuner.results_summary())

        #best_model = tuner.get_best_models()[0]
        early_stopping = EarlyStopping(monitor='val_cindex_metric_1', patience=30, mode='max')
        best_model = tuner.hypermodel.build(best_hps)
        history = best_model.fit([train_test_rna, train_test_clic], [train_test_event, train_test_time],
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 validation_data=([validation_rna, validation_clic], [validation_test_event, validation_time]),
                                 callbacks=[early_stopping])

        c_idx = best_model.evaluate([validation_rna, validation_clic], [validation_test_event, validation_time], verbose=0)
        print('cindex for fold_{} is : {}'.format(i, c_idx))

        # perform high risk and low risk analysis from the best model
        z_mean, z_std, z, pred_risk = best_model.encoder.predict([validation_rna, validation_clic])

        pred_risk_list.append(pred_risk)
        time.append(validation_time)
        event.append(validation_test_event)
        c_index.append(c_idx)

        print('loop validation c_index: ', c_index)
        i += 1

    return c_index, best_hyperparameters, pred_risk_list, shap_values, time, event, tuner_summary


c_index, best_hyperparameters, pred_risk_list, shap_values, time, event, tuner_summary = nested_cross_validate_model(pd.DataFrame(np.array(
    feature_dataset)), pd.DataFrame(np.array(clinical_features)), pd.DataFrame(np.array(y_event)), pd.DataFrame(np.array(y_survival_time)), n_folds=5)

print('mean c_index', np.mean(c_index))
print('std c_index', np.std(c_index))


# %%
# Save result as pickle
with open('Results/H-VAE-Coxsaved_Results.pkl', 'wb') as res:
    pickle.dump((
        c_index, best_hyperparameters, pred_risk_list, shap_values, time, event), res)


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
risk_group.to_csv('Results/GeneClinical-SurvivalRiskGroup.csv')
