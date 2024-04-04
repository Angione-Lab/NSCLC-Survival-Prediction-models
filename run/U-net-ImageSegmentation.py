# -*- coding: utf-8 -*-
"""

@author: suraj
"""

from pydicom import dcmread
from tensorflow.keras.layers import Input
import tensorflow.keras
import tensorflow as tf
import numpy as np
from sklearn.model_selection import KFold
import os
import matplotlib.pyplot as plt
import pickle
import pandas as pd

from utils.ImageProcessingUtils import ImagePreprocessing
from models.UnetModel import UnetModel

LOAD_SAVED_PICKLE = False
LOAD_WEIGHTS = False
IMG_SIZE_PX = 224
SLICE_COUNT = 20

patients = ''
if (LOAD_SAVED_PICKLE):
    data_dir = 'SavedObjects'
else:
    data_dir = 'Data/NSCLC_Images/'
    patients = os.listdir(data_dir)
    patients = sorted(patients, reverse=False)

# initialise ImageProcessing object
imagePreprocessing = ImagePreprocessing(IMG_SIZE_PX, IMG_SIZE_PX, SLICE_COUNT, data_dir)

# get list of labelled (segmented) patients
patient_list = imagePreprocessing.get_labelled_patients(patients)
# get list of unlabelled (unsegmented) patients
unsegmented_patients = [x for x in patients if x not in patient_list]

print(len(patients))
print(len(patient_list))
print(len(unsegmented_patients))

sorted(unsegmented_patients, reverse=False)

if (LOAD_SAVED_PICKLE):
    with open(data_dir+'/slices.pkl', 'rb') as f:
        slices = pickle.load(f)
    with open(data_dir+'/segments.pkl', 'rb') as f:
        segment_slices = pickle.load(f)
else:
    len(patient_list)
    train_img = imagePreprocessing.load_slices(patient_list)  # load slices
    segment = imagePreprocessing.load_segments(patient_list)  # load segments (labels)
    # Identifying segments with tumors
    segment_index = []
    segment_slices = []
    for idx, segment_slice in enumerate(segment):
        if (np.sum(segment_slice) >= 1):
            segment_slices.append(segment_slice)
            segment_index.append(idx)
    segment_slices = np.array(segment_slices)

    # getting tumorous slices
    slices = [x for i, x in enumerate(train_img) if i in segment_index]
    slices = np.array(slices)


"""Saving the slices and segments as pickle"""

if not LOAD_SAVED_PICKLE:
    with open('slices.pkl', 'wb') as slicespickle:
        pickle.dump(slices, slicespickle)

    with open('segments.pkl', 'wb') as segmentspickle:
        pickle.dump(segment_slices, segmentspickle)

# initialise U-Net model
umodel = UnetModel(IMG_SIZE_PX, IMG_SIZE_PX)
input_image = Input(shape=(IMG_SIZE_PX, IMG_SIZE_PX, 1))
segment_model = umodel.model(input_image)
segment_model.summary()


# for k fold validation
batch_size = 90
epochs = 100


# evaluate a model using k-fold cross-validation
def train_model(slices, segment_slices, n_folds=5):
    c_MeanIoU_metric, histories = list(), list()

    # inner fold
    i = 1
    k_inner_fold = KFold(n_folds, shuffle=True, random_state=5)
    for train_ix, test_ix in k_inner_fold.split(slices):
        print('fold- ', i)
        train_slices = slices[train_ix]
        test_slices = slices[test_ix]
        train_segment_slices = segment_slices[train_ix]
        test_segment_slices = segment_slices[test_ix]
        history = segment_model.fit(train_slices, train_segment_slices, validation_data=(
            test_slices, test_segment_slices), batch_size=10, epochs=50, shuffle=True)
        i = i+1

        MeanIoU = segment_model.evaluate(test_slices, test_segment_slices, verbose=0)

        histories.append(history)
        c_MeanIoU_metric.append(MeanIoU)

    return histories, c_MeanIoU_metric


histories = ''
c_MeanIoU_metric = ''
# %%
if (LOAD_WEIGHTS):
    segment_model.load_weights('SavedObjects/u_net_model.h5')

else:
    histories, c_MeanIoU_metric = train_model(slices, segment_slices, n_folds=5)
    print(np.mean(c_MeanIoU_metric, axis=0))
    print(np.std(c_MeanIoU_metric, axis=0))
    print(c_MeanIoU_metric)

# Read gene expression data to get common samples
RNA_PATH = 'Data/sorted_RNA_Seq.csv'
rna_df = pd.read_csv(RNA_PATH, index_col=0, header=0)
rna_df.sort_index(ascending=True, inplace=True)
print(rna_df.head())

# split samples in chunk for ROI extraction
common_samples = list(set(patients).intersection(rna_df.index))
print(len(common_samples))
phase1 = common_samples[:50]
print(phase1)
phase2 = common_samples[50:100]
print(phase2)
phase3 = common_samples[100:]
print(phase3)
len(np.concatenate((phase1, phase2, phase3)))

samples = phase1

imgs = imagePreprocessing.load_patientwise_slices(samples)


cropped_roi = []


def predict_segments():
    """
    Segment the tumour region and extract (Crop) the Region of Interest

    Returns
    -------
    TYPE
        DESCRIPTION. list of cropped tumour region from CT-scan slices.

    """
    tumour_segments = []
    for idx, x in enumerate(imgs):
        print(x.shape)
        pred = segment_model.predict(np.array(x))
        print(pred.shape)
        roi = imagePreprocessing.extract_ROI(x, pred)
        cropped_roi.append(roi)
    return np.array(cropped_roi)


crp_roi = predict_segments()

print(crp_roi.shape)
plt.imshow(tf.squeeze(crp_roi[0][:, :, 1]), cmap='gray')

# save the cropped ROI as pickle object
with open('chunked_cropped_images_part1.pkl', 'wb') as chunked_cropped_images:
    pickle.dump(crp_roi, chunked_cropped_images)


# %%
slice_path = 'D:/NSCLC_datasets/TCGA_Lung_dataset/TCGA-LUSC/TCGA-34-5928/'
imagePreprocessing = ImagePreprocessing(IMG_SIZE_PX, IMG_SIZE_PX, SLICE_COUNT, slice_path)
patients = os.listdir(slice_path)
imgs = imagePreprocessing.load_patientwise_slices(patients)
for idx, x in enumerate(imgs):
    print(x.shape)
    pred = segment_model.predict(np.array(x))
    print("predicted image:", pred.shape)
    roi = imagePreprocessing.extract_ROI(x, pred)
    # cropped_roi.append(roi)
    print(roi.shape)


slices = [dcmread(slice_path + '/' + s) for s in os.listdir(slice_path)]
slices.sort(key=lambda x: int(x.ImagePositionPatient[2]))
new_slices = []

for each_slice in slices:
    each_slice = each_slice.pixel_array
    import matplotlib.pyplot as plt
    plt.imshow(each_slice, cmap='gray')
    plt.show()
