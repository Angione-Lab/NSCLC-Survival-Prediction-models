# -*- coding: utf-8 -*-
"""
@author: SURAJ
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.compat.v1.keras.backend as K
from keras.regularizers import l2
from utils.SurvivalUtils import CoxPHLoss, CindexMetric


tf.config.run_functions_eagerly(True)  # used to show numpy values in Tensor object


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class X_VAE_GeneCox_model():

    def __init__(self, original_dim, clinical_dim, latent_dim):

        self.original_dim = original_dim
        self.clinical_dim = clinical_dim
        self.latent_dim = latent_dim

    def build_model(self, hp):

        learning_rate = hp.Float("lr", min_value=1e-7, max_value=1e-3, sampling="log")
        decay_rate = hp.Float("decay", min_value=1e-4, max_value=1e-1, sampling="log")
        l2_reg = hp.Float("l2_regularisation", min_value=1e-3, max_value=0.1)
        drop = hp.Choice("dropout", [0.8, 0.9])
        h1 = 1024
        h2 = 512

        encoder_inputs = keras.Input(shape=(self.original_dim,), name='encoder_input')
        x = layers.Dense(units=h1, activation="relu", kernel_initializer='he_uniform')(encoder_inputs)
        x = layers.Dense(units=h2, activation="relu", kernel_initializer='he_uniform')(x)
        x = layers.BatchNormalization()(x)
        z_mean = layers.Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])

        clinical_inputs = keras.Input(shape=(self.clinical_dim,), name='clinical_input')
        clinical_layer = layers.Dropout(drop)(clinical_inputs)
        clinical_layer = layers.Dense(units=4, activation="relu",
                                      kernel_initializer='he_uniform',
                                      name='clinical_layer_1',
                                      kernel_regularizer=l2(l2_reg))(clinical_layer)
        conc_clic_lat = layers.concatenate([clinical_layer, z_mean])
        c = layers.Dropout(drop)(conc_clic_lat)
        c = layers.Dense(units=32, activation="relu",
                         kernel_initializer='he_uniform',
                         kernel_regularizer=l2(l2_reg))(c)
        cox_outputs = layers.Dense(units=1, activation='sigmoid',
                                   use_bias=False, name='cox_hazard')(c)
        encoder = keras.Model([encoder_inputs, clinical_inputs], [z_mean, z_log_var, z, cox_outputs], name="encoder")
        tf.keras.utils.plot_model(encoder, show_shapes=True)

        """
        ## Build the decoder
        """

        latent_inputs = keras.Input(shape=(self.latent_dim,), name='decoder_input')
        x = layers.Dense(units=h2, activation='relu', kernel_initializer='he_uniform')(latent_inputs)
        x = layers.Dense(units=h1, activation='relu', kernel_initializer='he_uniform')(x)
        decoder_outputs = layers.Dense(self.original_dim, activation="sigmoid")(x)
        decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

        vae = G_XVAE(hp, encoder, decoder)
        adam = tf.keras.optimizers.Adam(learning_rate=learning_rate, decay=decay_rate)
        vae.compile(optimizer=adam)

        return vae


class G_XVAE(keras.Model):
    def __init__(self, hp, encoder, decoder, **kwargs):
        super(G_XVAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.cox_loss_tracker = keras.metrics.Mean(name="cox_loss")

        self.C_stop_iter = 1e5
        self.C_max = tf.constant(hp.Choice("c_stop", [25, 30, 35]), dtype=tf.float32)
        self.gamma = 1000
        self.kld_weight = 0.05
        self.num_iter = 0
        self.k_vl = 0.9
        self.k_cl = 1

        self.loss_fn = CoxPHLoss()
        self.val_cindex_metric = CindexMetric()

    @property
    def metrics(self):
        return [
            self.val_cindex_metric
        ]

    def train_step(self, data):
        self.num_iter += 1
        with tf.GradientTape() as tape:

            rna, clinical = data[0]
            y_event, y_time = data[1]
            z_mean, z_log_var, z, pred_risk = self.encoder([rna, clinical])
            reconstruction = self.decoder(z)
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            reconstruction_loss = keras.metrics.mean_squared_error(rna, reconstruction)
            cox_loss = self.loss_fn(y_true=[y_event, y_time], y_pred=pred_risk)
            cox_regularisation_loss = 0
            if len(self.encoder.losses) > 0:
                cox_regularisation_loss = tf.math.add_n(self.encoder.losses)
            C = tf.clip_by_value(self.C_max/self.C_stop_iter * self.num_iter, 0, self.C_max.numpy())  # https://arxiv.org/pdf/1804.03599.pdf
            total_loss = (reconstruction_loss + self.gamma * self.kld_weight * K.abs(kl_loss - C)) * \
                self.k_vl + (cox_loss + cox_regularisation_loss) * self.k_cl

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.cox_loss_tracker.update_state(cox_loss)
        self.val_cindex_metric.update_state(y_time, y_event, pred_risk)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "cox_loss": self.cox_loss_tracker.result(),
            "cindex": self.val_cindex_metric.result()
        }

    def test_step(self, data):
        rna, clinical = data[0]
        y_event, y_time = data[1]
        z_mean, z_log_var, z, pred_risk = self.encoder([rna, clinical], training=False)
        self.val_cindex_metric.update_state(y_time, y_event, pred_risk)
        return {m.name: m.result() for m in self.metrics}
