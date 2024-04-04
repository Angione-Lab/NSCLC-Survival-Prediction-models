# -*- coding: utf-8 -*-
"""
@author: SURAJ
"""

import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
import matplotlib.pyplot as plt
from utils.SurvivalUtils import km_plot
import keras_tuner as kt
from keras.callbacks import EarlyStopping
from utils.DataLoader import load_data
from models.ClinicalCoxmodel import Clinical_Cox_model

from sklearn.preprocessing import MinMaxScaler

tf.config.run_functions_eagerly(True)

img, rnaseq_scaled_df, pathway_mask, clinical_features, y_event, y_survival_time = load_data('X-img-gene-clic')

clinical_dim = clinical_features.shape[1]


"""# K-Fold cross validation to fit the model
"""
batch_size = 30
epochs = 100
outer_fold = 5
MAX_TRIALS = 2
EXECUTION_PER_TRIAL = 1
search_epochs = 20

SEED = 20


# evaluate a model using k-fold cross-validation
def nested_cross_validate_model(clinical_features, y_event, y_time):
    val_c_index, best_hyperparameters, shap_values, pred_risk_list, time, event, high_risk_rna_list, high_risk_img_list, high_risk_clinical_list = list(
    ), list(), list(), list(), list(), list(), list(), list(), list()

    # prepare cross validation
    k_outer_fold = KFold(outer_fold, shuffle=True, random_state=SEED)
    # enumerate splits
    i = 1
    # outer fold
    for train_test_ix, validation_idx in k_outer_fold.split(clinical_features):

        # initialise X_VAE_Cox_model for Radiogenomics data
        clinical_model = Clinical_Cox_model(clinical_dim)

        train_test_clic = clinical_features.iloc[train_test_ix]
        validation_clic = clinical_features.iloc[validation_idx]
        train_test_event = y_event[train_test_ix]
        validation_test_event = y_event[validation_idx]
        train_test_time = y_time[train_test_ix]
        validation_time = y_time[validation_idx]

        train_test_clic = pd.DataFrame(MinMaxScaler().fit_transform(train_test_clic), index=train_test_clic.index, columns=train_test_clic.columns)
        validation_clic = pd.DataFrame(MinMaxScaler().fit_transform(validation_clic), index=validation_clic.index, columns=validation_clic.columns)

        # inner fold
        tuner = kt.BayesianOptimization(
            hypermodel=clinical_model.build_model,
            objective=kt.Objective("val_cindex", direction="max"),
            max_trials=MAX_TRIALS,
            overwrite=True,
            executions_per_trial=EXECUTION_PER_TRIAL,
            directory="Clinical-hyper_dir",
            project_name='Clinical-Cox{}'.format(i),
        )

        # stratified train test split
        train_clic, test_clic, train_event, test_event, train_time, test_time = \
            train_test_split(train_test_clic, train_test_event, train_test_time,  test_size=0.2,
                             stratify=train_test_event, shuffle=True, random_state=SEED)

        early_stopping = EarlyStopping(monitor='val_cindex', patience=50,  mode='max')

        tuner.search(train_clic, [train_event, train_time],
                     batch_size=batch_size,
                     epochs=search_epochs,
                     validation_data=(test_clic, [test_event, test_time]),
                     callbacks=[early_stopping]
                     )
        best_hps = tuner.get_best_hyperparameters(1)[0]
        best_hyperparameters.append(best_hps.values)
        print(best_hps.values)

        # best_model = tuner.get_best_models(1)[0]
        best_model = tuner.hypermodel.build(best_hps)  # rebuild the model using best hyperparameter
        checkpoint_path = 'best_savedmodel/Hvaecoxmodel_{}'.format(i)
        # checkpoint = CustomCheckpoint(filepath=checkpoint_path,model = best_model,  monitor='val_cindex',  save_best_only=True, mode='max', patience=10)
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=False,
            monitor='val_cindex',
            mode='max',
            save_best_only=True)

        history = best_model.fit(train_test_clic, [train_test_event, train_test_time],
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 validation_data=(validation_clic, [validation_test_event, validation_time]),
                                 callbacks=[early_stopping]
                                 )
        cp_cidx = best_model.evaluate(validation_clic, [validation_test_event, validation_time], verbose=0)

        print('\n cindex for fold_{} is :')
        print('Val_cindex: ', cp_cidx[1])

        plt.text(0.2, 0.4, 'val_cindex: {}'.format(cp_cidx[1]), fontsize=10)
        plt.show()

        # perform high risk and low risk analysis from the best model
        pred_risk = best_model.clinical_model.predict(validation_clic)
        median_risk = np.median(pred_risk)

        Risk = np.where(pred_risk > median_risk, 1, 0)

        validation_clic['Hazard'] = pred_risk
        high_risk_clinical = validation_clic[validation_clic['Hazard'] > median_risk]
        high_risk_clinical.drop('Hazard', axis=1, inplace=True)

        high_risk_clinical_list.append(high_risk_clinical)
        pred_risk_list.append(pred_risk)
        time.append(validation_time)
        event.append(validation_test_event)

        val_c_index.append({"validation_set": cp_cidx[1]})

        # release memory to avoid OOM exception
        del tuner
        del history
        tf.keras.backend.clear_session()

        i = i+1
    return val_c_index, best_hyperparameters, shap_values, pred_risk_list, time, event, high_risk_clinical_list


val_c_index, best_hyperparameters, shap_values, pred_risk_list, time, event, high_risk_clinical_list \
    = nested_cross_validate_model(clinical_features, np.array(y_event), np.array(y_survival_time))

print(pd.DataFrame(val_c_index))

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
risk_group.to_csv('Results/only_clinical-Risk-Group.csv')
