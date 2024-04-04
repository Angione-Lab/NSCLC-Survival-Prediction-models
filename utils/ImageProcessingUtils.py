# -*- coding: utf-8 -*-
"""
@author: suraj
"""

import tensorflow.keras
import tensorflow as tf
import numpy as np
import os
import math
import cv2
import matplotlib.pyplot as plt
from pydicom import dcmread


class ImagePreprocessing():
    def __init__(self, img_size_x, img_size_y, hm_slices, data_dir):
        self.img_size_x = img_size_x
        self.img_size_y = img_size_y
        self.data_dir = data_dir
        self.HM_SLICES = hm_slices

    def load_slices(self, patient_list):
        """
        data generator to load CT scan image sices

        Parameters
        ----------

        patient_list : TYPE
            DESCRIPTION. list of patients

        Yields
        ------
        each_slice : TYPE
            DESCRIPTION.

        """
        patient_images = []
        for patient in patient_list:
            path = self.data_dir + patient
            #path = data_dir + 'R01-082'
            slice_path = path + '/CT_Images/'
            slices = [dcmread(slice_path + '/' + s) for s in os.listdir(slice_path)]
            slices.sort(key=lambda x: int(x.ImagePositionPatient[2]))
            #new_slices = []
            print(patient, '-->', len(slices))
            for each_slice in slices:
                each_slice = each_slice.pixel_array

                # standardization
                #each_slice = (each_slice - each_slice.mean())/each_slice.std()
                # each_slice.astype("float32")
                each_slice = each_slice.astype('float32') / np.max(each_slice)
                each_slice = cv2.resize(np.array(each_slice), (self.img_size_x, self.img_size_y))
                each_slice = np.expand_dims(each_slice, axis=-1)
                yield each_slice

    def load_segments(self, patient_list):
        """
        data generator to load segment sices

        Parameters
        ----------
        img_size_x : TYPE
            DESCRIPTION. image dimension x
        img_size_y : TYPE
            DESCRIPTION. image dimension y
        data_dir : TYPE
            DESCRIPTION. image path
        patient_list : TYPE
            DESCRIPTION. list of patients

        Yields
        ------
        each_slice : TYPE
            DESCRIPTION.

        """
        patient_images = []
        for patient in patient_list:

            path = self.data_dir + patient
            #path = data_dir + 'R01-082'
            segment_path = path + '/segmentation/'
            segment = [dcmread(segment_path + '/' + s) for s in os.listdir(segment_path)]
            #segment.sort(key = lambda x: int(x.ImagePositionPatient[2]))
            #img = segment[0].pixel_array
            # plt.imshow(img[100,:,:])

            for each_slice in segment:
                each_slice = each_slice.pixel_array
               # print(each_slice.shape)#
                print(patient, '-->', each_slice.shape)
                for img_slice in range(each_slice.shape[0]):
                    img_s = each_slice[img_slice, :, :]
                    img_s = cv2.resize(img_s, (self.img_size_x, self.img_size_y))
                    img_s = np.expand_dims(img_s, axis=-1)
                    yield img_s

    def get_labelled_patients(self, patients):
        """
        Get list of labelled (segmented) patients 

        Parameters
        ----------
        patients : TYPE
            DESCRIPTION. list of patients (name of folders)

        Returns
        -------
        filtered_patients : TYPE
            DESCRIPTION. List of Patients

        """
        filtered_patients = []
        for patient in patients:

            path = self.data_dir + patient
            slice_path = path + '/CT_Images/'
            slices = [dcmread(slice_path + '/' + s) for s in os.listdir(slice_path)]
            segment_path = path + '/segmentation/'
            segment = [dcmread(segment_path + '/' + s) for s in os.listdir(segment_path)]
            if (len(slices) == segment[0].pixel_array.shape[0]):
                filtered_patients.append(patient)

        return filtered_patients

    def load_patientwise_slices(self, samples):
        """
        Get list of slices grouped by patients

        Parameters
        ----------
        samples : TYPE
            DESCRIPTION. list of patients

        Yields
        ------
        TYPE
            DESCRIPTION. list of slices grouped by patients.

        """
        patient_images = []
        for patient in samples:
            path = self.data_dir + patient
            slice_path = path + '/CT_Images/'
            slices = [dcmread(slice_path + '/' + s) for s in os.listdir(slice_path)]
            slices.sort(key=lambda x: int(x.ImagePositionPatient[2]))
            new_slices = []
            print(patient, '-->', len(slices))
            for each_slice in slices:
                each_slice = each_slice.pixel_array

                # standardization
                # each_slice = (each_slice - each_slice.mean())/each_slice.std()
                each_slice = each_slice.astype('float32') / np.max(each_slice)
                # each_slice.astype("float32")
                each_slice = cv2.resize(np.array(each_slice), (self.img_size_x, self.img_size_y))
                each_slice = np.expand_dims(each_slice, axis=-1)
                new_slices.append(each_slice)
            patient_images.append(new_slices)
            yield np.array(new_slices)

    def chunks(self, l, n):
        print("Chunking :", len(l), n)
        for i in range(0, len(l), n):
            yield l[i:i + n]

    def mean(self, l):
        return sum(l) / len(l)

    def visualise_seg(self, segment, new_img):
        plt.figure()
        f, ax = plt.subplots(ncols=2, figsize=(8, 8))
        ax[0].imshow((tf.squeeze(segment)), cmap='gray')
        ax[0].set_title("Segment")
        ax[0].axis('off')

        ax[1].imshow((tf.squeeze(new_img)), cmap='gray')
        ax[1].set_title("ROI")
        ax[1].axis('off')
        plt.show()

    def crop_extracted_roi(self, patient_slices, patient_segments):
        cropped_images = []
        new_slices = []
        for i in range(patient_slices.shape[0]):
            image = patient_slices[i]
            segment = patient_segments[i]
            img = cv2.resize(image, (150, 150), interpolation=cv2.INTER_AREA)
            segment = cv2.resize(segment, (150, 150), interpolation=cv2.INTER_AREA).astype(np.uint8)
            cnts, hierachy = cv2.findContours(segment.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            idx = 0
            #cropped_images = []
            for c in cnts:
                x, y, w, h = cv2.boundingRect(c)
                print(h, w)
                if w >= 2 and h >= 2:
                    x_pos = x-25 if x-25 > 0 else 0
                    y_pos = y-25 if y-25 > 0 else 0

                    new_img = img[y_pos:y+h+25, x_pos:x+w+25]
                    self.visualise_seg(segment, new_img)
                    cropped_image = cv2.resize(new_img, (224, 224), interpolation=cv2.INTER_AREA)
                    cropped_image = np.expand_dims(cropped_image, axis=-1)
                    cropped_images.append(cropped_image)

        conc_img = []
        print('cropped_images: ', len(cropped_images))

        chunk_sizes = math.ceil(len(cropped_images) / self.HM_SLICES)
        for slice_chunk in self.chunks(cropped_images, chunk_sizes):
            slice_chunk = list(map(self.mean, zip(*slice_chunk)))
            new_slices.append(slice_chunk)

        if len(new_slices) == self.HM_SLICES-1:
            new_slices.append(new_slices[-1])

        if len(new_slices) == self.HM_SLICES-2:
            new_slices.append(new_slices[-1])
            new_slices.append(new_slices[-1])

        if len(new_slices) == self.HM_SLICES-3:
            new_slices.append(new_slices[-1])
            new_slices.append(new_slices[-1])
            new_slices.append(new_slices[-1])

        if len(new_slices) == self.HM_SLICES-4:
            new_slices.append(new_slices[-1])
            new_slices.append(new_slices[-1])
            new_slices.append(new_slices[-1])
            new_slices.append(new_slices[-1])

        if len(new_slices) == self.HM_SLICES+2:
            new_val = list(map(self.mean, zip(*[new_slices[self.HM_SLICES-1], new_slices[self.HM_SLICES],])))
            del new_slices[self.HM_SLICES]
            new_slices[self.HM_SLICES-1] = new_val

        if len(new_slices) == self.HM_SLICES+1:
            new_val = list(map(self.mean, zip(*[new_slices[self.HM_SLICES-1], new_slices[self.HM_SLICES],])))
            del new_slices[self.HM_SLICES]
            new_slices[self.HM_SLICES-1] = new_val
        if (len(new_slices) > 0):
            conc_img = np.concatenate(new_slices, axis=-1)
        print('conc_img', conc_img.shape)

        return np.array(conc_img)

    def extract_ROI(self, slices, segment):
        # Identifying segments with tumors
        segment_index = []
        segment_slices = []
        for idx, segment_slice in enumerate(segment):
            if (np.sum(segment_slice) >= 1):
                segment_slices.append(segment_slice)
                segment_index.append(idx)
        segment_slices = np.array(segment_slices)

        # getting tumorous slices
        slices = [x for i, x in enumerate(slices) if i in segment_index]
        slices = np.array(slices)
        cropped_roi = self.crop_extracted_roi(slices, segment_slices)
        slices = []
        return cropped_roi
