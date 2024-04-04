"""
@author: SURAJ
"""

import tensorflow.keras as keras
import os
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
import shap
import pickle
import matplotlib.pyplot as plt
from utils.SurvivalUtils import km_plot, x_vae_shap_values, compute_x_vae_mm_score, x_vae_gradcam_heatmap
import keras_tuner as kt
from keras.callbacks import EarlyStopping
from utils.DataLoader import load_data
from models.XVAECoxmodel import X_VAE_Cox_model, Sampling
from sklearn.preprocessing import MinMaxScaler

tf.config.run_functions_eagerly(True)

img, rnaseq_scaled_df, pathway_mask, clinical_features, y_event, y_survival_time = load_data(
    'X-img-gene-clic')
luad_img, luad_rnaseq_scaled_df, pathway_mask, luad_clinical_features, luad_y_event, luad_y_survival_time = load_data(
    'X-TCGA-LUAD')
lusc_img, lusc_rnaseq_scaled_df, pathway_mask, lusc_clinical_features, lusc_y_event, lusc_y_survival_time = load_data(
    'X-TCGA-LUSC')

# Normalise OOD dataset
luad_rnaseq_scaled_df = MinMaxScaler().fit_transform(luad_rnaseq_scaled_df)
lusc_rnaseq_scaled_df = MinMaxScaler().fit_transform(lusc_rnaseq_scaled_df)
luad_clinical_features = MinMaxScaler().fit_transform(luad_clinical_features)
lusc_clinical_features = MinMaxScaler().fit_transform(lusc_clinical_features)

original_gene_dim = rnaseq_scaled_df.shape[1]
pathway_shape = pathway_mask.shape[1]
clinical_dim = clinical_features.shape[1]


"""# K-Fold cross validation to fit the model
"""
batch_size = 26
epochs = 150
outer_fold = 5
MAX_TRIALS = 2
EXECUTION_PER_TRIAL = 1
search_epochs = 20

SEED = 20

# initialise X_VAE_Cox_model for Radiogenomics data
# x_vae_model = X_VAE_Cox_model(original_gene_dim, pathway_shape, clinical_dim, pathway_mask)


def plot_label_clusters(z_mean, risk):
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=risk, s=100)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.show()


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
        plt.imshow(de[0][i].reshape(224, 224, 5)[:, :, 2])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


# evaluate a model using k-fold cross-validation
def nested_cross_validate_model(img, rnaseq_scaled_df, clinical_features, y_event, y_time):
    val_c_index, best_model_list, best_hyperparameters, shap_values, pred_risk_list, time, event, high_risk_rna_list, high_risk_img_list, high_risk_clinical_list = list(
    ), list(), list(), list(), list(), list(), list(), list(), list(), list()

    # prepare cross validation
    k_outer_fold = KFold(outer_fold, shuffle=True, random_state=SEED)
    # enumerate splits
    i = 1
    # outer fold
    for train_test_ix, validation_idx in k_outer_fold.split(rnaseq_scaled_df):

        # initialise X_VAE_Cox_model for Radiogenomics data
        x_vae_model = X_VAE_Cox_model(
            original_gene_dim, pathway_shape, clinical_dim, pathway_mask)

        train_test_img = img[train_test_ix]
        validation_img = img[validation_idx]
        train_test_rna = rnaseq_scaled_df.iloc[train_test_ix]
        validation_rna = rnaseq_scaled_df.iloc[validation_idx]
        train_test_clic = clinical_features.iloc[train_test_ix]
        validation_clic = clinical_features.iloc[validation_idx]
        train_test_event = y_event.iloc[train_test_ix]
        validation_test_event = y_event.iloc[validation_idx]
        train_test_time = y_time.iloc[train_test_ix]
        validation_time = y_time.iloc[validation_idx]

        train_test_rna = MinMaxScaler().fit_transform(train_test_rna)
        validation_rna = MinMaxScaler().fit_transform(validation_rna)
        train_test_clic = MinMaxScaler().fit_transform(train_test_clic)
        validation_clic = MinMaxScaler().fit_transform(validation_clic)

        # inner fold
        tuner = kt.BayesianOptimization(
            hypermodel=x_vae_model.build_model,
            objective=kt.Objective("val_cindex", direction="max"),
            max_trials=MAX_TRIALS,
            overwrite=True,
            executions_per_trial=EXECUTION_PER_TRIAL,
            directory="X-VAE-hyper_dir",
            project_name='X-VAE-Cox{}'.format(i),
        )

        # stratified train test split
        train_img, test_img, train_rna, test_rna, train_clic, test_clic, train_event, test_event, train_time, test_time = \
            train_test_split(train_test_img, train_test_rna, train_test_clic, train_test_event, train_test_time,  test_size=0.2,
                             stratify=train_test_event, shuffle=True, random_state=SEED)

        early_stopping = EarlyStopping(
            monitor='val_cindex', patience=50,  mode='max')

        tuner.search([train_img, train_rna, train_clic], [train_event, train_time],
                     batch_size=batch_size,
                     epochs=search_epochs,
                     validation_data=([test_img, test_rna, test_clic], [
                                      test_event, test_time]),
                     # validation_data=([lusc_img, lusc_rnaseq_scaled_df, lusc_clinical_features], [lusc_y_event, lusc_y_survival_time]),
                     callbacks=[early_stopping]
                     )

        best_hps = tuner.get_best_hyperparameters(1)[0]
        best_hyperparameters.append(best_hps.values)
        print(best_hps.values)

        # best_model = tuner.get_best_models(1)[0]
        # rebuild the model using best hyperparameter
        best_model = tuner.hypermodel.build(best_hps)
        checkpoint_path = 'best_savedmodel/Hvaecoxmodel_{}'.format(i)
        # checkpoint = CustomCheckpoint(filepath=checkpoint_path,model = best_model,  monitor='val_cindex',  save_best_only=True, mode='max', patience=10)
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=False,
            monitor='val_cindex',
            mode='max',
            save_best_only=True)

        history = best_model.fit([train_test_img, train_test_rna, train_test_clic], [train_test_event, train_test_time],
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 validation_data=([validation_img, validation_rna, validation_clic], [
                                                  validation_test_event, validation_time]),
                                 # , model_checkpoint_callback]
                                 callbacks=[early_stopping]
                                 )
        # saved_encoder = load_model(checkpoint_path+'/encoder')
        # saved_decoder = load_model(checkpoint_path+'/decoder')
        # cp_cidx = x_vae_model.evaluate_model([validation_img, validation_rna,validation_clic,validation_test_event,validation_time],best_model)
        cp_cidx = best_model.evaluate([validation_img, validation_rna, validation_clic], [
                                      validation_test_event, validation_time], verbose=0)
        lusc_cp_cidx = best_model.evaluate([lusc_img, lusc_rnaseq_scaled_df, lusc_clinical_features], [
                                           lusc_y_event, lusc_y_survival_time], verbose=0)
        luad_cp_cidx = best_model.evaluate([luad_img, luad_rnaseq_scaled_df, luad_clinical_features], [
                                           luad_y_event, luad_y_survival_time], verbose=0)

        # tcga_cp_cidx = best_model.evaluate([tcga_img, tcga_rna, tcga_cli], [tcga_y_event, tcga_y_survival_time], verbose=0)

        print('\n cindex for fold_{} is :')
        print('Val_cindex: ', cp_cidx[1], 'luad_cindex: ',
              luad_cp_cidx[1], 'lusc_cindex: ', lusc_cp_cidx[1])

        # for key in list(history.history.keys()):
        #    plt.plot(history.history[key])
        #    plt.legend([key, '_fold_{}'.format(i)], loc='upper right')
        #    plt.text(0.5, 0.4, 'val_cindex: {}'.format(cp_cidx[1]), fontsize=10)
        #    plt.show()

        plt.text(0.2, 0.4, 'val_cindex: {}, \n luad_cindex: {} \n lusc_cindex: {}'.format(
            cp_cidx[1], luad_cp_cidx[1], lusc_cp_cidx[1]), fontsize=10)
        plt.show()

        # perform high risk and low risk analysis from the best model
        z_mean, z_std, z, pred_risk = best_model.encoder.predict(
            [validation_img, validation_rna, validation_clic])
        de = best_model.decoder.predict(z)

        # best_model.encoder.save('SavedObjects/X-vae/X_vae_encoder_{}.h5'.format(i))
        # best_model.decoder.save('SavedObjects/X-vae/X_vae_decoder_{}.h5'.format(i))

        reconstructed_image_plot(validation_img, de)

        median_risk = np.median(pred_risk)

        Risk = np.where(pred_risk > median_risk, 1, 0)
        # plot_label_clusters(z_mean, Risk)

        validation_rna_df = pd.DataFrame(validation_rna)
        validation_rna_df['Hazard'] = pred_risk
        high_risk_rna = validation_rna_df[validation_rna_df['Hazard'] > median_risk]
        high_risk_img = validation_img[high_risk_rna.index]
        high_risk_clinical = validation_clic[high_risk_rna.index]
        validation_rna_df.drop('Hazard', axis=1, inplace=True)
        high_risk_rna.drop('Hazard', axis=1, inplace=True)

        # SHAP values for high risk samples
        shap_values.append(x_vae_shap_values(best_model, [train_test_img, train_test_rna, train_test_clic], [
                           high_risk_img, np.array(high_risk_rna), high_risk_clinical]))

        high_risk_rna_list.append(high_risk_rna)
        high_risk_img_list.append(high_risk_img)
        high_risk_clinical_list.append(high_risk_clinical)
        pred_risk_list.append(pred_risk)
        time.append(validation_time)
        event.append(validation_test_event)
        # val_c_index.append(cp_cidx[1])
        val_c_index.append({"validation_set": cp_cidx[1],
                           "LUAD_set": luad_cp_cidx[1],
                            "LUSC_set": lusc_cp_cidx[1]})
        #                   "TCGA_set": tcga_cp_cidx})
        best_model_list.append(best_model)

        # release memory to avoid OOM exception
        del tuner
        del history
        del x_vae_model
        del best_model
        tf.keras.backend.clear_session()

        i = i+1
    return val_c_index, best_model_list, best_hyperparameters, shap_values, pred_risk_list, time, event, high_risk_rna_list, high_risk_img_list, high_risk_clinical_list


val_c_index, best_model_list, best_hyperparameters, shap_values, pred_risk_list, time, event, high_risk_rna_list, high_risk_img_list, high_risk_clinical_list = nested_cross_validate_model(
    img, rnaseq_scaled_df, clinical_features, y_event, y_survival_time)

# print('mean c_index', np.mean(val_c_index))
# print('std c_index', np.std(val_c_index))
print(pd.DataFrame(val_c_index))

# %%
for i in range(len(best_model_list)):
    print(best_model_list[i].encoder.summary())
    best_model_list[i].encoder.save('Results_Saved_models/X_vae_encoder_good_grad_{}.h5'.format(i))
    best_model_list[i].decoder.save('Results_Saved_models/X_vae_decoder_good_grad_{}.h5'.format(i))

with open('Results_Saved_models/att_X_Vae_results_good_grad.pkl', 'wb') as res:
    pickle.dump((
        val_c_index, best_hyperparameters, shap_values, pred_risk_list, time, event, high_risk_rna_list, high_risk_img_list, high_risk_clinical_list), res)

# %%

hazard_list = np.array(pred_risk_list).reshape(-1)
time_list = np.array(time).reshape(-1)
event_list = np.array(event).reshape(-1)

km_plot(pd.DataFrame(event_list, columns=['Survival Status']), pd.DataFrame(
    time_list, columns=['survival_time']), hazard_list)

median_risk = np.median(hazard_list)
df = pd.DataFrame()
df['Hazard'] = np.array(hazard_list)
df['Risk'] = np.where(df.Hazard > median_risk, 1, 0)
risk_group = pd.concat([pd.DataFrame(event_list, columns=['Survival Status']), pd.DataFrame(
    time_list, columns=['survival_time']), df['Hazard'], df['Risk']], axis=1)

time_df = pd.concat(time)
risk_group.index = time_df.index

risk_group.to_csv('Results_Saved_models/Att_X-VAE-Survival-Risk-Group_for_DEG.csv')

# %%

"""
SHAP interpretation

"""

score = compute_x_vae_mm_score(shap_values)
print(score)


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
    gene_sha = pd.concat([gene_sha, pd.DataFrame(shap_obj[i][0][0][1])])

# append high risk RNA smaples
rna_obj = high_risk_rna_list
rna = []
rna = pd.DataFrame(rna_obj[0])
for i in range(1, 5):
    rna = pd.concat([rna, pd.DataFrame(rna_obj[i])])

rna.columns = rnaseq_scaled_df.columns
gene_sha.columns = rnaseq_scaled_df.columns

# shap for gene expression
shap_val_Exp = Explanation(gene_sha, rna)
shap.summary_plot(shap_val_Exp, max_display=40, show=False)
plt.rcParams['font.size'] = '20'
# plt.savefig('Results_Saved_models/att_x-vae-shap_rna.pdf')


# Get list of SHAP importance for genes and save in csv for further biological interpretation
feature_names = rnaseq_scaled_df.columns
vals = np.abs(gene_sha.values).mean(0)
shap_importance = pd.DataFrame(list(zip(feature_names, vals)),
                               columns=['col_name', 'feature_importance_vals'])
shap_importance.sort_values(by=['feature_importance_vals'],
                            ascending=False, inplace=True)
shap_importance.head()
shap_importance.to_csv('Results_Saved_models/att_XVAE-GeneShapImportance.csv')

# %%
# clinical shap values

clinical_shap = pd.DataFrame()
for i in range(5):
    clinical_shap = clinical_shap.append(pd.DataFrame(shap_obj[i][0][0][2]))

highrisk_clinical = pd.DataFrame()
for i in range(5):
    highrisk_clinical = highrisk_clinical.append(
        pd.DataFrame(high_risk_clinical_list[i]))

clinical_shap.columns = clinical_features.columns
highrisk_clinical.columns = clinical_features.columns


shap_val_Exp = Explanation(clinical_shap, highrisk_clinical)
shap.summary_plot(shap_val_Exp, max_display=40, show=False)
plt.rcParams['font.size'] = '20'
plt.savefig('Results_Saved_models/att_x-vae-shap_clinical.pdf')


# %%
""" Image Shap values """
for k in range(1):  # 5 fold
    list_id = k
    image_shap_value = shap_obj[list_id][0][0][0]
    high_risk_images = high_risk_img_list[list_id]

    for j in range(5):  # 5 slices of images
        shap_img = np.array(
            [np.expand_dims(image_shap_value[i][:, :, j], axis=-1) for i in range(len(image_shap_value))])
        test_img = np.array(
            [np.expand_dims(high_risk_images[i][:, :, j], axis=-1) for i in range(len(image_shap_value))])
        shap.image_plot(shap_img, test_img, show=False)
        # plt.savefig('Results_Saved_models/SHAP_images/att_shap_img_plots_fold_{}_samp_{}_slice_{}.pdf'.format(k, i, j))

# %%
b_mode = best_model_list[0]

pth_layer = b_mode.encoder.get_layer('Pathway_Layer').get_weights()
pth_layer = pd.DataFrame(pth_layer[1], index=pathway_mask.columns)

att_layer = b_mode.encoder.get_layer('Pathway_attention').get_weights()[7]
pth_att = pd.DataFrame(att_layer, index=pathway_mask.columns)
pth_att.to_csv('pth_att5_final.csv')

# %%

with open('Results/X_VAE_cox/att_X_Vae_results.pkl', 'rb') as f:
    data = pickle.load(f)
    print(data)

# %%
"Grad-CAM plot for image modality"

fold_id = 0
model = best_model_list[fold_id].encoder  # keras.models.load_model('SavedObjects/X-vae/X_vae_encoder_1.h5', custom_objects={"Sampling": Sampling})

image = high_risk_img_list[fold_id]
rna = np.array(high_risk_rna_list[fold_id])
clic = np.array(high_risk_clinical_list[fold_id])

plt.figure()
f, ax = plt.subplots(nrows=clic.shape[0], ncols=10, figsize=(30, 30))
slices, ht_img = x_vae_gradcam_heatmap(model, image, rna, clic, alpha=0.7)
for i in range(clic.shape[0]):
    for j in range(len(ht_img[i])):
        ax[i, j].imshow(ht_img[i][j])
        ax[i, j].axis('off')

        ax[i, j+len(ht_img[i])].imshow(slices[i][j], cmap='gray')
        ax[i, j+len(ht_img[i])].axis('off')
# plt.savefig('Results_Saved_models/GradCam_images/run2_Gradcam_plot_fold_{}.pdf'.format(fold_id))
