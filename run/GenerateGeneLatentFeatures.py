# -*- coding: utf-8 -*-
"""
@author: SURAJ
"""

import pickle
import random
import numpy as np
import keras_tuner as kt
import pandas as pd
from sklearn.model_selection import KFold
from utils.SurvivalUtils import km_plot
from sklearn.model_selection import train_test_split
import tensorflow as tf
from utils.DataLoader import load_data
from keras.callbacks import EarlyStopping
from models.GenePathwayAutoencoder import GeneCox_model
from sklearn.preprocessing import MinMaxScaler


"""# K-Fold cross validation to fit the model
"""
SEED = 20
random.seed(SEED)
batch_size = 26
search_epochs = 20
epochs = 100
max_trial = 3
exec_per_trial = 1
rnaseq_scaled_df, pathway_mask, y_event, y_survival_time = load_data('gene-pathwaymask')
lusc_rnaseq_df, pathway_mask, lusc_y_event, lusc_y_survival_time = load_data('lusc_gene-pathwaymask')
luad_rnaseq_df, pathway_mask, luad_y_event, luad_y_survival_time = load_data('luad_gene-pathwaymask')

lusc_rnaseq_df = pd.DataFrame(MinMaxScaler().fit_transform(lusc_rnaseq_df), index=lusc_rnaseq_df.index)
luad_rnaseq_df = pd.DataFrame(MinMaxScaler().fit_transform(luad_rnaseq_df), index=luad_rnaseq_df.index)

latent_dim = 500
original_dim = rnaseq_scaled_df.shape[1]

with tf.device('/GPU:0'):
    genecox_model = GeneCox_model(original_dim, latent_dim, pathway_mask)

    # evaluate a model using k-fold cross-validation
    def nested_cross_validate_model(feature_df, y_event, y_time, n_folds=5):
        c_index, best_model_list, pred_list, latent_feature, time, event, luad_latent_features, lusc_latent_features = list(
        ), list(), list(), list(), list(), list(), list(), list()

        sample_id = feature_df.index
        feature_df.reset_index(drop=True, inplace=True)
        y_event.reset_index(drop=True, inplace=True)
        y_time.reset_index(drop=True, inplace=True)

        # prepare cross validation
        k_outer_fold = KFold(n_folds, shuffle=True, random_state=SEED)
        # enumerate splits
        i = 1
        # outer fold
        for train_test_ix, validation_idx in k_outer_fold.split(feature_df):

            train_test_rna = feature_df.loc[train_test_ix].reset_index(drop=True)
            validation_rna = feature_df.loc[validation_idx].reset_index(drop=True)
            train_test_event = y_event.loc[train_test_ix].reset_index(drop=True)
            validation_test_event = y_event.loc[validation_idx].reset_index(drop=True)
            train_test_time = y_time.loc[train_test_ix].reset_index(drop=True)
            validation_time = y_time.loc[validation_idx].reset_index(drop=True)
            val_sample_id = sample_id[validation_idx]

            train_test_rna = MinMaxScaler().fit_transform(train_test_rna)
            validation_rna = MinMaxScaler().fit_transform(validation_rna)
            print(train_test_rna)

            tuner = kt.BayesianOptimization(
                hypermodel=genecox_model.build_model,
                objective=kt.Objective("val_cindex_metric", direction="max"),
                max_trials=max_trial,
                overwrite=True,
                executions_per_trial=exec_per_trial,
                directory="G_XVAE-hyper_dir",
                project_name='Gene_XVAE-Cox{}'.format(i),
            )
            # stratified train test split
            train_rna, test_rna, train_event, test_event, train_time, test_time = \
                train_test_split(train_test_rna, train_test_event, train_test_time,  test_size=0.15,
                                 stratify=train_test_event, shuffle=True, random_state=SEED)

            tuner.search(train_rna, [train_event, train_time],
                         batch_size=batch_size,
                         epochs=search_epochs,
                         validation_data=(test_rna, [test_event, test_time]))

            best_hps = tuner.get_best_hyperparameters(1)[0]
            # best_hyperparameters.append(best_hps.values)
            print(best_hps.values)
            print(tuner.results_summary())
            with tf.device('/CPU:0'):
                # best_model = tuner.get_best_models()[0]
                early_stopping = EarlyStopping(monitor='val_cindex_metric_1', patience=30, min_delta=0.0001,  mode='max')
                best_model = tuner.hypermodel.build(best_hps)
                history = best_model.fit(train_test_rna, [train_test_event, train_test_time],
                                         batch_size=batch_size,
                                         epochs=epochs,
                                         validation_data=(validation_rna, [validation_test_event, validation_time]),
                                         callbacks=[early_stopping])

            c_idx = best_model.evaluate(validation_rna, [validation_test_event, validation_time], verbose=0)
            print('cindex for fold_{} is : {}'.format(i, c_idx))

            # perform high risk and low risk analysis from the best model
            z, pred_risk = best_model.encoder.predict(validation_rna)

            #  generate latent features of LUAD and LUSC images for OOD validation
            luad_z, luad_pred_risk = best_model.encoder.predict(luad_rnaseq_df)
            lusc_z, lusc_pred_risk = best_model.encoder.predict(lusc_rnaseq_df)

            # best_model.encoder.save('SavedObjects/H_VAE/GeneSparseEncoder_{}.h5'.format(i))
            # best_model.decoder.save('SavedObjects/H_VAE/GeneSparseDecoder_{}.h5'.format(i))
            lat_feat = pd.DataFrame(z, index=val_sample_id)
            luad_lat_feat = pd.DataFrame(luad_z, index=luad_rnaseq_df.index)
            lusc_lat_feat = pd.DataFrame(lusc_z, index=lusc_rnaseq_df.index)

            c_index.append(c_idx)
            pred_list.append(pred_risk)
            time.append(validation_time)
            event.append(validation_test_event)
            latent_feature.append(lat_feat)

            luad_latent_features.append(luad_lat_feat)
            lusc_latent_features.append(lusc_lat_feat)

            best_model_list.append(best_model)
            print('loop validation c_index: ', c_index)
            i += 1

        return c_index, latent_feature, pred_list, time, event, best_model_list, luad_latent_features, lusc_latent_features

    c_index, latent_feature, pred_risk_list, time, event, best_model_list, luad_latent_features, lusc_latent_features = nested_cross_validate_model(
        rnaseq_scaled_df, y_event, y_survival_time, n_folds=5)

print('mean c_index', np.mean(c_index))
print('std c_index', np.std(c_index))
# %%
for i in range(len(best_model_list)):
    best_model_list[i].encoder.save('SavedObjects/H_VAE/GeneSparseEncoder_{}.h5'.format(i))
    best_model_list[i].decoder.save('SavedObjects/H_VAE/GeneSparseDecoder_{}.h5'.format(i))


# %%

lusc_z, pred_risk = best_model_list[4].encoder.predict(lusc_rnaseq_df)
lusc_lat_feat = pd.DataFrame(lusc_z, index=lusc_rnaseq_df.index)
lusc_lat_feat.to_csv('Results/tcga_lusc_cox_gene_latent.csv')

luad_z, pred_risk = best_model_list[4].encoder.predict(luad_rnaseq_df)
luad_lat_feat = pd.DataFrame(luad_z, index=luad_rnaseq_df.index)
luad_lat_feat.to_csv('Results/tcga_luad_cox_gene_latent.csv')


# %%
lat_feature_df = pd.DataFrame(columns=latent_feature[0].columns)
for lat in latent_feature:
    print(type(lat))
    lat_feature_df = pd.concat([lat_feature_df, lat])

lat_feature_df.sort_index().to_csv('Results/cox_gene_latent.csv')

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
risk_group.to_csv('Results/GeneOnlySurvivalRiskGroup.csv')
