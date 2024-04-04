# -*- coding: utf-8 -*-
"""
@author: SURAJ
"""

import keras_tuner as kt
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
import pickle
from utils.SurvivalUtils import km_plot
from models.ImgClinicalCoxmodel import X_VAE_ImgCox_model
from utils.DataLoader import load_data
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
import random
feature_dataset, clinical_features, y_event, y_survival_time = load_data('image_clinical')


latent_dim = 256
clinical_dim = clinical_features.shape[1]


"""# K-Fold cross validation to fit the model
"""
batch_size = 26
epochs = 100
search_epochs = 20
max_trial = 3
exec_per_trial = 1
seed = 20
random.seed(20)


x_vae_imgcox_model = X_VAE_ImgCox_model(clinical_dim, latent_dim)
# evaluate a model using k-fold cross-validation


def nested_cross_validate_model(feature_df, clinical_features, y_event, y_time, n_folds=5):
    val_c_index, best_hyperparameters, pred_risk_list, time, event = list(), list(), list(), list(), list()

    y_event = y_event.reset_index(drop=True)
    # prepare cross validation
    k_outer_fold = KFold(n_folds, shuffle=True, random_state=seed)
    # enumerate splits
    i = 1
    # outer fold
    for train_test_ix, validation_idx in k_outer_fold.split(feature_df):

        train_test_rad = feature_df[train_test_ix]
        validation_rad = feature_df[validation_idx]
        train_test_clic = clinical_features.loc[train_test_ix].reset_index(drop=True)
        validation_clic = clinical_features.loc[validation_idx].reset_index(drop=True)
        train_test_event = y_event.loc[train_test_ix].reset_index(drop=True)
        validation_test_event = y_event.loc[validation_idx].reset_index(drop=True)
        train_test_time = y_time.loc[train_test_ix].reset_index(drop=True)
        validation_time = y_time.loc[validation_idx].reset_index(drop=True)

        train_test_clic = pd.DataFrame(MinMaxScaler().fit_transform(train_test_clic))
        validation_clic = pd.DataFrame(MinMaxScaler().fit_transform(validation_clic))

        tuner = kt.BayesianOptimization(
            hypermodel=x_vae_imgcox_model.build_model,
            objective=kt.Objective("val_cindex", direction="max"),
            max_trials=max_trial,
            overwrite=True,
            executions_per_trial=exec_per_trial,
            directory="Img_XVAE-hyper_dir",
            project_name='Img_XVAE-Cox{}'.format(i),
        )
        # stratified train test split
        train_rad, test_rad, train_clic, test_clic, train_event, test_event, train_time, test_time = \
            train_test_split(train_test_rad, train_test_clic, train_test_event, train_test_time,  test_size=0.2,
                             stratify=train_test_event, shuffle=True, random_state=seed)

        earlystopper = EarlyStopping(monitor='val_cindex', mode='max', patience=30, verbose=1)
        tuner.search([train_rad, train_clic], [train_event, train_time],
                     batch_size=batch_size,
                     epochs=search_epochs,
                     validation_data=([test_rad, test_clic], [test_event, test_time]))

        best_hps = tuner.get_best_hyperparameters(1)[0]
        best_hyperparameters.append(best_hps.values)
        print(best_hps.values)

        #best_model = tuner.get_best_models()[0]
        best_model = tuner.hypermodel.build(best_hps)
        history = best_model.fit([train_test_rad, train_test_clic], [train_test_event, train_test_time],
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 validation_data=([validation_rad, validation_clic], [validation_test_event, validation_time]),
                                 callbacks=[earlystopper])

        val_c_idx = best_model.evaluate([validation_rad, validation_clic], [validation_test_event, validation_time], verbose=0)
        print('val cindex for fold_{} is : {}'.format(i, val_c_idx))
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
        plt.text(0.5, 0.5, 'val_cindex: {}'.format(val_c_idx[1]), fontsize=10)
        plt.show()

        # perform high risk and low risk analysis from the best model
        z, pred_risk = best_model.encoder.predict([validation_rad, validation_clic])

        pred_risk_list.append(pred_risk)
        time.append(validation_time)
        event.append(validation_test_event)
        val_c_index.append(val_c_idx[1])
        i = i+1
    return val_c_index, best_hyperparameters, pred_risk_list, time, event


val_c_index, best_hyperparameters, pred_risk_list, time, event = nested_cross_validate_model(
    feature_dataset, clinical_features, y_event, y_survival_time, n_folds=5)
print(val_c_index)
print('mean c_index', np.mean(val_c_index))
print('std c_index', np.std(val_c_index))

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
risk_group.to_csv('Results/ImageClinicalSurvivalRiskGroup.csv')
