# -*- coding: utf-8 -*-
"""
@author: suraj
"""

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import initializers
from keras.regularizers import l2
from utils.SurvivalUtils import CoxPHLoss, CindexMetric
tf.config.run_functions_eagerly(True)


learning_rate = 1e-3
decay = 0.01
# scheduled learning rate
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=learning_rate, decay_steps=10000, decay_rate=decay)


class ImgCox_model():

    def __init__(self, latent_dim):

        self.latent_dim = latent_dim

    def build_model(self, hp):

        # hyperparameters
        n1_unit = hp.Choice('n1', [64, 32], default=64)
        n2_unit = hp.Choice('n2', [64, 32, 16], default=32)
        l2_regularizer = hp.Float("l2_regularisation", min_value=1e-3, max_value=0.1)
        drop = hp.Choice("dropout", [0.8, 0.9])

        """
        ## Build the encoder
        """

        encoder_inputs = keras.Input(shape=(224, 224, 5))
        x = layers.Conv2D(128, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Flatten()(x)
        x = layers.BatchNormalization()(x)
        z = layers.Dense(self.latent_dim,  activation="tanh")(x)

        c = layers.Dropout(drop)(z)
        c = layers.Dense(n2_unit, activation="tanh", kernel_initializer=initializers.GlorotUniform(
            seed=None), kernel_regularizer=l2(l2_regularizer))(c)
        c = layers.BatchNormalization()(c)
        cox_outputs = layers.Dense(1, use_bias=False, kernel_regularizer=l2(l2_regularizer), name='cox_hazard')(c)
        encoder = keras.Model(encoder_inputs, [z, cox_outputs], name="encoder")
        encoder.summary()

        """
        ## Build the decoder
        """

        latent_inputs = keras.Input(shape=(self.latent_dim,))
        x = layers.Dense(56 * 56 * 64, activation="relu")(latent_inputs)
        x = layers.Reshape((56, 56, 64))(x)
        x = layers.Conv2DTranspose(128, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
        decoder_outputs = layers.Conv2DTranspose(5, 3, activation="sigmoid", padding="same")(x)
        decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
        decoder.summary()

        #tf.keras.utils.plot_model(cox_network, to_file = 'cox_network.png', show_shapes=True)

        vae = I_AE(hp, encoder, decoder)
        adam = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        vae.compile(optimizer=adam)

        return vae


class I_AE(keras.Model):
    def __init__(self, hp, encoder, decoder, **kwargs):
        super(I_AE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.cox_loss_tracker = keras.metrics.Mean(name="cox_loss")

        self.loss_fn = CoxPHLoss()
        self.val_cindex_metric = CindexMetric()

    @property
    def metrics(self):
        return [
            self.val_cindex_metric,
            self.cox_loss_tracker
        ]

    # train step

    def train_step(self, data):
        with tf.GradientTape() as tape:

            image = data[0]
            y_event, y_time = data[1]
            z, pred_risk = self.encoder(image)
            reconstruction = self.decoder(z)
            mse = tf.keras.losses.MeanSquaredError()
            reconstruction_loss = mse(image, reconstruction)
            cox_loss = self.loss_fn(y_true=[y_event, y_time], y_pred=pred_risk)
            regularisation_loss = 0
            if len(self.encoder.losses) > 0:
                regularisation_loss += tf.math.add_n(self.encoder.losses)
            total_loss = reconstruction_loss + cox_loss + regularisation_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.cox_loss_tracker.update_state(cox_loss)
        self.val_cindex_metric.update_state(y_time, y_event, pred_risk)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "cox_loss": self.cox_loss_tracker.result(),
            "cindex": self.val_cindex_metric.result()
        }

    # test step
    def test_step(self, data):
        image = data[0]
        y_event, y_time = data[1]
        z, pred_risk = self.encoder(image, training=False)
        reconstruction = self.decoder(z, training=False)
        mse = tf.keras.losses.MeanSquaredError()
        reconstruction_loss = mse(image, reconstruction)
        cox_loss = self.loss_fn(y_true=[y_event, y_time], y_pred=pred_risk)
        regularisation_loss = 0
        if len(self.encoder.losses) > 0:
            regularisation_loss += tf.math.add_n(self.encoder.losses)
        total_loss = reconstruction_loss + cox_loss + regularisation_loss

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.cox_loss_tracker.update_state(cox_loss)
        self.val_cindex_metric.update_state(y_time, y_event, pred_risk)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "cox_loss": self.cox_loss_tracker.result(),
            "cindex": self.val_cindex_metric.result()
        }
