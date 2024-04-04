# -*- coding: utf-8 -*-
"""
@author: SURAJ
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.compat.v1.keras.backend as K
from keras.regularizers import l2
from tensorflow.keras import initializers
from utils.SurvivalUtils import CoxPHLoss, CindexMetric


tf.config.run_functions_eagerly(True)  # used to show numpy values in Tensor object


learning_rate = 1e-3
decay = 0.1

# scheduled learning rate
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=learning_rate, decay_steps=10000, decay_rate=decay)


class GeneCox_model():

    def __init__(self, original_dim, latent_dim, pathway_mask):

        self.original_dim = original_dim
        self.latent_dim = latent_dim
        self.pathway_mask = pathway_mask

    def build_model(self, hp):

        l2_reg = hp.Float("l2_regularisation", min_value=1e-3, max_value=0.1)
        drop = hp.Choice("dropout", [0.8, 0.9])
        n1_unit = hp.Choice('n2', [32, 16], default=32)

        """
        ## Build the encoder
        """

        encoder_inputs = keras.Input(shape=(self.original_dim,), name='encoder_input')
        x = layers.BatchNormalization()(encoder_inputs)
        x = layers.Dense(self.original_dim, activation="relu", kernel_initializer='glorot_uniform')(encoder_inputs)
        x = layers.Dense(self.pathway_mask.shape[1], activation="relu", kernel_initializer='glorot_uniform', name='Pathway_Layer')(x)
        x = layers.BatchNormalization()(x)
        z = layers.Dense(self.latent_dim,  activation="tanh")(x)

        """
        ## Build the cox layer
        """

        c = layers.Dropout(drop)(z)
        c = layers.Dense(n1_unit, activation="tanh", kernel_initializer=initializers.GlorotUniform(seed=None), kernel_regularizer=l2(l2_reg))(c)
        c = layers.BatchNormalization()(c)
        cox_outputs = layers.Dense(1, use_bias=False, kernel_regularizer=l2(l2_reg), name='cox_hazard')(c)
        encoder = keras.Model(encoder_inputs, [z, cox_outputs], name="encoder")
        encoder.summary()

        """
        ## Build the decoder
        """

        latent_inputs = keras.Input(shape=(self.latent_dim,), name='decoder_input')
        x = layers.Dense(self.pathway_mask.shape[1], activation="relu", kernel_initializer='glorot_uniform', name='pathway_decoder')(latent_inputs)
        x = layers.Dense(self.original_dim, activation="relu", kernel_initializer='glorot_uniform', name='gene_decoder')(x)
        decoder_outputs = layers.Dense(self.original_dim, activation="sigmoid")(x)
        decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

        vae = G_XVAE(hp, encoder, decoder, self.pathway_mask)
        adam = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        vae.compile(optimizer=adam)

        return vae


class G_XVAE(keras.Model):
    def __init__(self, hp, encoder, decoder, pathway_mask, **kwargs):
        super(G_XVAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.pathway_mask = pathway_mask
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
            self.val_cindex_metric
        ]

    def update_weights(self, layer, pathway_mask):
        old_weight = layer.get_weights()
        new_weight = layer.get_weights()[0]*pathway_mask
        updated_weight = old_weight
        updated_weight[0] = new_weight
        layer.set_weights(updated_weight)

# train steps
    def train_step(self, data):
        with tf.GradientTape() as tape:

            rna = data[0]
            y_event, y_time = data[1]
            z, pred_risk = self.encoder(rna)
            reconstruction = self.decoder(z)
            reconstruction_loss = keras.metrics.mean_squared_error(rna, reconstruction)
            cox_loss = self.loss_fn(y_true=[y_event, y_time], y_pred=pred_risk)
            cox_regularisation_loss = 0
            if len(self.encoder.losses) > 0:
                cox_regularisation_loss = tf.math.add_n(self.encoder.losses)
            total_loss = reconstruction_loss + cox_loss + cox_regularisation_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.update_weights(self.encoder.get_layer('Pathway_Layer'), self. pathway_mask)
        self.update_weights(self.decoder.get_layer('gene_decoder'), self.pathway_mask.T)
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

# test steps
    def test_step(self, data):
        rna = data[0]
        y_event, y_time = data[1]
        z, pred_risk = self.encoder(rna, training=False)
        self.val_cindex_metric.update_state(y_time, y_event, pred_risk)
        return {m.name: m.result() for m in self.metrics}
