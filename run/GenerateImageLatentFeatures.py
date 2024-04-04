# -*- coding: utf-8 -*-
"""
@author: Suraj
"""

import pickle
import keras_tuner as kt
from sklearn.model_selection import KFold, train_test_split
from utils.SurvivalUtils import km_plot
from utils.DataLoader import load_data
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
from models.ImageAutoencoder import ImgCox_model

feature_dataset, clinical_features, y_event, y_survival_time = load_data('image_clinical')
luad_feature_dataset, luad_clinical_features, luad_y_event, luad_y_survival_time = load_data('TCGA_LUAD_image_clinical')
lusc_feature_dataset, lusc_clinical_features, lusc_y_event, lusc_y_survival_time = load_data('TCGA_LUSC_image_clinical')


latent_dim = 500
batch_size = 13
epochs = 100
search_epochs = 20
max_trial = 3
exec_per_trial = 1

imgcox_model = ImgCox_model(latent_dim)


def reconstructed_image_plot(img, de):
    n = 5
    plt.figure(figsize=(10, 4))
    for i in range(n):
        # Display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(img[i].reshape(224, 224, 5)[:, :, 2])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(de[i].reshape(224, 224, 5)[:, :, 2])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


# evaluate a model using k-fold cross-validation
def nested_cross_validate_model(radio_genomics_df, y_event, y_time, n_folds=5):
    val_c_index, latent_features, pred_risk_list, time, event, luad_latent_features, lusc_latent_features = list(), list(), list(), list(), list(), list(), list()

    sample_id = y_event.index
    y_event = y_event.reset_index(drop=True)
    # prepare cross validation
    k_outer_fold = KFold(n_folds, shuffle=True, random_state=10)
    # enumerate splits
    i = 1
    # outer fold
    for train_test_ix, validation_idx in k_outer_fold.split(radio_genomics_df):

        train_test_rad = radio_genomics_df[train_test_ix]
        validation_rad = radio_genomics_df[validation_idx]
        train_test_event = y_event.loc[train_test_ix].reset_index(drop=True)
        validation_test_event = y_event.loc[validation_idx].reset_index(drop=True)
        train_test_time = y_time.loc[train_test_ix].reset_index(drop=True)
        validation_time = y_time.loc[validation_idx].reset_index(drop=True)
        val_sample_id = sample_id[validation_idx]

        tuner = kt.BayesianOptimization(
            hypermodel=imgcox_model.build_model,
            objective=kt.Objective("val_cindex", direction="max"),
            max_trials=max_trial,
            overwrite=True,
            executions_per_trial=exec_per_trial,
            directory="Img_XVAE-hyper_dir",
            project_name='Img_XVAE-Cox{}'.format(i),
        )
        # stratified train test split
        train_rad, test_rad, train_event, test_event, train_time, test_time = \
            train_test_split(train_test_rad, train_test_event, train_test_time,  test_size=0.2,
                             stratify=train_test_event, shuffle=True, random_state=10)

        earlystopper = EarlyStopping(monitor='val_cindex', mode='max', patience=30, verbose=1)
        tuner.search(train_rad, [train_event, train_time],
                     batch_size=batch_size,
                     epochs=search_epochs,
                     validation_data=(test_rad, [test_event, test_time]))

        best_hps = tuner.get_best_hyperparameters(1)[0]
        print(best_hps.values)

        #best_model = tuner.get_best_models()[0]
        best_model = tuner.hypermodel.build(best_hps)
        history = best_model.fit(train_test_rad, [train_test_event, train_test_time],
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 validation_data=(validation_rad, [validation_test_event, validation_time]),
                                 # callbacks=[earlystopper]
                                 )

        val_c_idx = best_model.evaluate(validation_rad, [validation_test_event, validation_time], verbose=0)
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
        z, pred_risk = best_model.encoder.predict(validation_rad)

        #  generate latent features of LUAD and LUSC images for OOD validation
        luad_z, luad_pred_risk = best_model.encoder.predict(luad_feature_dataset)
        lusc_z, lusc_pred_risk = best_model.encoder.predict(lusc_feature_dataset)
        # best_model.encoder.save('SavedObjects/H_VAE/ImageEncoder_{}.h5'.format(i))
        # best_model.decoder.save('SavedObjects/H_VAE/Image Decoder_{}.h5'.format(i))
        lat_feat = pd.DataFrame(z, index=val_sample_id)
        luad_lat_feat = pd.DataFrame(luad_z, index=luad_clinical_features.index)
        lusc_lat_feat = pd.DataFrame(lusc_z, index=lusc_clinical_features.index)

        de = best_model.decoder.predict(z)
        print(de.shape)
        print(z.shape)
        reconstructed_image_plot(validation_rad, de)

        pred_risk_list.append(pred_risk)
        time.append(validation_time)
        event.append(validation_test_event)
        val_c_index.append(val_c_idx[1])
        latent_features.append(lat_feat)
        luad_latent_features.append(luad_lat_feat)
        lusc_latent_features.append(lusc_lat_feat)
        i = i+1
    return val_c_index, latent_features, pred_risk_list, time, event, luad_latent_features, lusc_latent_features


val_c_index, latent_features, pred_risk_list, time, event, luad_latent_features, lusc_latent_features = nested_cross_validate_model(
    feature_dataset, y_event, y_survival_time, n_folds=5)


print(val_c_index)
print('mean c_index', np.mean(val_c_index))
print('std c_index', np.std(val_c_index))


# %%
luad_latent_features[3].sort_index().to_csv('Results/tcga_luad_cox_image_latent.csv')
lusc_latent_features[3].sort_index().to_csv('Results/tcga_lusc_cox_image_latent.csv')

with open('SavedObjects/tcga_nsclc_image_latent_features.pkl', 'wb') as res:
    pickle.dump((luad_latent_features, lusc_latent_features, latent_features), res)

# %%
lat_feature_df = pd.DataFrame(columns=latent_features[0].columns)
for lat in latent_features:
    print(type(lat))
    lat_feature_df = pd.concat([lat_feature_df, lat])

lat_feature_df.sort_index().to_csv('Results/cox_image_latent.csv')

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
risk_group.to_csv('Results/ImageOnlySurvivalRiskGroup.csv')
