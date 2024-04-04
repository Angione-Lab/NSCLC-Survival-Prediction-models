# -*- coding: utf-8 -*-
"""
@author: suraj
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.regularizers import l2
import tensorflow.compat.v1.keras.backend as K
from utils.SurvivalUtils import CoxPHLoss, CindexMetric
from tensorflow.keras import initializers
import random
import numpy as np

tf.config.run_functions_eagerly(True)  # used to show numpy values in Tensor object
tf.data.experimental.enable_debug_mode()
random.seed(20)


"""
## Create a sampling layer
"""


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class H_VAE_Cox_model():

    def __init__(self, original_dim, clinical_dim):

        self.original_dim = original_dim
        self.clinical_dim = clinical_dim
        #self.latent_dim   = latent_dim

    def build_model(self, hp):

        l2_regularizer = hp.Float("l2_regularisation", min_value=0.005, max_value=0.1)
        drop = hp.Choice("dropout", [0.8, 0.9])
        n1_unit = hp.Choice('n1', [32, 16], default=32)
        n2_unit = hp.Choice('n2', [16], default=16)
        h1 = 256
        h2 = 256
        self.latent_dim = hp.Choice('latent', [128, 256], default=128)

        learning_rate = 1e-3
        decay = 0.1
        # scheduled learning rate
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=learning_rate, decay_steps=10000, decay_rate=decay)

        """
        ## Build the encoder
        """

        encoder_inputs = keras.Input(shape=(self.original_dim,), name='encoder_input')
        clinical_inputs = keras.Input(shape=(self.clinical_dim,), name='clinical_input')
        #clinical_outputs = layers.Dense(8, activation="relu")(clinical_inputs)
        x = layers.Dense(units=h1, activation="relu",
                         kernel_initializer='he_uniform',
                         bias_initializer='he_uniform')(encoder_inputs)
        #x = layers.BatchNormalization()(x)
        x = layers.Dense(units=h2, activation="relu",
                         kernel_initializer='he_uniform',
                         bias_initializer='he_uniform')(x)
        x = layers.BatchNormalization()(x)
        z_mean = layers.Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])

        c = layers.concatenate([clinical_inputs, z_mean])
        c = layers.BatchNormalization()(c)
        c = layers.Dense(n1_unit, activation="tanh", kernel_initializer=initializers.GlorotUniform(
            seed=None), kernel_regularizer=l2(l2_regularizer))(c)
        c = layers.Dropout(drop)(c)
        c = layers.Dense(n2_unit, activation="tanh", kernel_initializer=initializers.GlorotUniform(
            seed=None), kernel_regularizer=l2(l2_regularizer))(c)
        cox_outputs = layers.Dense(1, use_bias=False, kernel_regularizer=l2(l2_regularizer), name='cox_hazard')(c)
        encoder = keras.Model([encoder_inputs, clinical_inputs], [z_mean, z_log_var, z, cox_outputs], name="encoder")

        encoder.summary()
        """
        ## Build the decoder
        """

        latent_inputs = keras.Input(shape=(self.latent_dim,), name='decoder_input')
        x = layers.Dense(units=h2, activation='relu',
                         kernel_initializer='he_uniform')(latent_inputs)
        x = layers.Dense(units=h1, activation='relu',
                         kernel_initializer='he_uniform')(x)
        decoder_outputs = layers.Dense(self.original_dim, activation="sigmoid")(x)
        decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

        vae = VAE(hp, encoder, decoder)
        adam = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        vae.compile(optimizer=adam)

        return vae


class VAE(keras.Model):

    def __init__(self, hp, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )

        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.cox_loss_tracker = keras.metrics.Mean(name="cox_loss")

        self.C_stop_iter = 1e5
        self.C_max = tf.constant(hp.Choice("c_stop", [30, 35]), dtype=tf.float32)
        self.gamma = 1000
        self.kld_weight = 0.05
        self.num_iter = 0

        self.k_vl = 1
        self.k_cl = 1

        self.loss_fn = CoxPHLoss()
        self.val_cindex_metric = CindexMetric()

    @property
    def metrics(self):
        return [
            self.val_cindex_metric
        ]
    # train steps

    def train_step(self, data):
        self.num_iter += 1
        regularisation_loss = 0
        with tf.GradientTape() as tape:

            rna, clinical = data[0]
            y_event, y_time = data[1]
            z_mean, z_log_var, z, pred_risk = self.encoder([rna, clinical])
            reconstruction = self.decoder(z)
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            reconstruction_loss = keras.metrics.mean_squared_error(rna, reconstruction)
            cox_loss = self.loss_fn(y_true=[y_event, y_time], y_pred=pred_risk)
            C = tf.clip_by_value(self.C_max/self.C_stop_iter * self.num_iter, 0, self.C_max.numpy())  # https://arxiv.org/pdf/1804.03599.pdf

            if len(self.encoder.losses) > 0:
                regularisation_loss = tf.math.add_n(self.encoder.losses)

            total_loss = reconstruction_loss * self.k_vl + (self.gamma * self.kld_weight *
                                                            K.abs(kl_loss - C) + cox_loss + regularisation_loss) * self.k_cl

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
    # test steps

    def test_step(self, data):
        rna, clinical = data[0]
        y_event, y_time = data[1]
        z_mean, z_log_var, z, pred_risk = self.encoder([rna, clinical], training=False)
        reconstruction = self.decoder(z, training=False)
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        reconstruction_loss = keras.metrics.mean_squared_error(rna, reconstruction)
        cox_loss = self.loss_fn(y_true=[y_event, y_time], y_pred=pred_risk)
        C = tf.clip_by_value(self.C_max/self.C_stop_iter * self.num_iter, 0, self.C_max.numpy())  # https://arxiv.org/pdf/1804.03599.pdf

        if len(self.encoder.losses) > 0:
            regularisation_loss = tf.math.add_n(self.encoder.losses)

        total_loss = reconstruction_loss + self.gamma * self.kld_weight * K.abs(kl_loss - C) + cox_loss + regularisation_loss
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
