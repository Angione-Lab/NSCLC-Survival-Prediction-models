# -*- coding: utf-8 -*-
"""
@author: SURAJ
"""

from tensorflow.keras.layers import concatenate, Conv2DTranspose, BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
import tensorflow.keras
import tensorflow as tf
import numpy as np


class UnetModel():
    def __init__(self, sample_width, sample_height, lr=1e-3):
        self.sample_width = sample_width
        self.sample_height = sample_height
        self.lr = lr
        self.smooth = 1.

    # K.set_image_data_format('channels_first')
    def mean_iou(y_true, y_pred):
        prec = []
        for t in np.arange(0.5, 1.0, 0.05):
            y_pred_ = tf.cast((y_pred > t), tf.int32)
            score, up_opt = tf.compat.v1.metrics.mean_iou(y_true, y_pred_, 2)
            K.get_session().run(tf.local_variables_initializer())
            with tf.control_dependencies([up_opt]):
                score = tf.identity(score)
            prec.append(score)
        return K.mean(K.stack(prec), axis=0)

    def dice_coef(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + self.smooth) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + self.smooth)

    def dice_coef_loss(self, y_true, y_pred):
        return 1 - self.dice_coef(y_true, y_pred)

    def model(self, input_img, TRAINABLE=False):

        base_model = VGG16(weights='imagenet')

        for layer in base_model.layers:
            layer.trainable = TRAINABLE

    # -------------------encoder----------------------------

    #    block1
        print(input_img)
        l1 = Conv2D(3, (1, 1))(input_img)  # map N channels data to 3 channels
        conv1 = base_model.get_layer('block1_conv1')(l1)
        conv1 = base_model.get_layer('block1_conv2')(conv1)
        pool1 = base_model.get_layer('block1_pool')(conv1)

    #    block2
        conv2 = base_model.get_layer('block2_conv1')(pool1)
        conv2 = base_model.get_layer('block2_conv2')(conv2)
        pool2 = base_model.get_layer('block2_pool')(conv2)

    #    block3
        conv3 = base_model.get_layer('block3_conv1')(pool2)
        conv3 = base_model.get_layer('block3_conv2')(conv3)
        conv3 = base_model.get_layer('block3_conv3')(conv3)
        pool3 = base_model.get_layer('block3_pool')(conv3)

    #    block4
        conv4 = base_model.get_layer('block4_conv1')(pool3)
        conv4 = base_model.get_layer('block4_conv2')(conv4)
        conv4 = base_model.get_layer('block4_conv3')(conv4)
        pool4 = base_model.get_layer('block4_pool')(conv4)

    #    block5
        """conv5=base_model.get_layer('block5_conv1')(pool4)
        conv5=base_model.get_layer('block5_conv2')(conv5)
        conv5=base_model.get_layer('block5_conv3')(conv5)"""

        conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same',  name='btl_conv')(pool4)
        batch5 = BatchNormalization(axis=-1)(conv5)
        conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same',  name='btl_conv_1')(batch5)
        batch5 = BatchNormalization(axis=-1)(conv5)

        up6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same', name='up6')(batch5)
        up6 = concatenate([up6, conv4], axis=-1,  name='conc_6_4')
        conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(up6)
        batch6 = BatchNormalization(axis=-1)(conv6)
        conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(batch6)
        batch6 = BatchNormalization(axis=-1)(conv6)

        up7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(batch6)
        up7 = concatenate([up7, conv3], axis=-1,  name='conc_7_3')
        conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(up7)
        batch7 = BatchNormalization(axis=-1)(conv7)
        conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(batch7)
        batch7 = BatchNormalization(axis=-1)(conv7)

        up8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(batch7)
        up8 = concatenate([up8, conv2], axis=-1,  name='conc_8_2')
        conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(up8)
        batch8 = BatchNormalization(axis=-1)(conv8)
        conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(batch8)
        batch8 = BatchNormalization(axis=-1)(conv8)

        up9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(batch8)
        up9 = concatenate([up9, conv1], axis=-1,  name='conc_9_1')
        conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(up9)
        batch9 = BatchNormalization(axis=-1)(conv9)
        conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(batch9)
        batch9 = BatchNormalization(axis=-1)(conv9)

        conv10 = Conv2D(1, (1, 1), activation='sigmoid')(batch9)

        model = Model(inputs=[input_img], outputs=[conv10])
        lr_schedule = tensorflow.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-2, decay_steps=10000, decay_rate=0.9)

        model.compile(optimizer=Adam(learning_rate=lr_schedule), loss=self.dice_coef_loss, metrics=[tf.keras.metrics.MeanIoU(num_classes=2)])

        return model
