# -*- coding: utf-8 -*-
"""
@author: SURAJ
"""

import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow.compat.v1.keras.backend as K
from tensorflow.keras.regularizers import l2
from utils.SurvivalUtils import CoxPHLoss, CindexMetric, cindex
import keras_tuner as kt
import numpy as np
from tensorflow.keras import initializers

tf.config.run_functions_eagerly(True)


Capacity_max_iter = 1e5
gamma = 1000.
max_capacity = 25
kld_weight = 0.005

learning_rate = 1e-3
decay = 0.1

# scheduled learning rate
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=learning_rate, decay_steps=50, decay_rate=decay)


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


class X_VAE_Cox_model(kt.HyperModel):

    def __init__(self, original_gene_dim, pathway_shape, clinical_dim, pathway_mask):

        self.original_gene_dim = original_gene_dim
        self.pathway_shape = pathway_shape
        self.clinical_dim = clinical_dim
        self.pathway_mask = pathway_mask

    def build_model(self, hp):

        l2_regularizer = hp.Float(
            "l2_regularisation", min_value=1e-3, max_value=1e-1, sampling="log")
        drop = hp.Choice('dropout', [0.8, 0.9])
        n1_unit = hp.Choice('n1', [128, 64, 32], default=128)
        self.latent_dim = hp.Choice('latent', [256, 350], default=350)
        num_heads = hp.Choice('num_heads', [2, 4], default=4)

        """
        Encode
        """

        vgg_model = keras.applications.VGG19(
            weights='imagenet', include_top=False, pooling='avg')

        encoder_img_inputs = keras.Input(shape=(224, 224, 5), name="image_input")
        x = layers.Conv2D(3, (3, 3), padding='same', activation='relu',)(encoder_img_inputs)

        x = layers.MultiHeadAttention(key_dim=5, num_heads=1, dropout=drop,
                                      name='encoder_img_attention',
                                      use_bias=False, attention_axes=0)(x, x)

        for i, layer in enumerate(vgg_model.layers):
            if i != 0 and i < 5:
                layer.trainable = False
                x = layer(x)

        # x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same", name="encoder_img_conv_layer")(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Flatten()(x)
        x = layers.Dropout(drop)(x)
        encoder_img = layers.Dense(self.pathway_shape, activation="relu",
                                   kernel_initializer='glorot_uniform', name='encoder_img_layer_3')(x)

        # layer_1
        encoder_gene_inputs = keras.Input(
            shape=(self.original_gene_dim,), name="gene_input")
        clinical_inputs = keras.Input(
            shape=(self.clinical_dim,), name='clinical_input')

        # layer 2
        encoder_gene = layers.Dense(self.pathway_shape,
                                    activation="relu",
                                    kernel_initializer='glorot_uniform',
                                    name='Pathway_Layer')(encoder_gene_inputs)

        encoder_gene = layers.MultiHeadAttention(key_dim=self.pathway_shape,
                                                 num_heads=num_heads,
                                                 dropout=drop,
                                                 use_bias=True,
                                                 name='Pathway_attention',
                                                 attention_axes=0)(encoder_gene, encoder_gene)

        encoder_gene = layers.Dropout(drop)(encoder_gene)
        encoder_gene = layers.LayerNormalization(epsilon=1e-6)(encoder_gene)

        # layer_4
        encoder_img_gene = layers.MultiHeadAttention(key_dim=self.pathway_shape,
                                                     num_heads=num_heads,
                                                     dropout=drop,
                                                     use_bias=True,
                                                     name='encoder_img_gene_attention',
                                                     attention_axes=0)(encoder_img, encoder_gene)

        encoder_gene_img = layers.MultiHeadAttention(key_dim=self.pathway_shape,
                                                     num_heads=num_heads,
                                                     dropout=drop,
                                                     use_bias=True,
                                                     name='encoder_gene_img_attention',
                                                     attention_axes=0)(encoder_gene, encoder_img)

        encoder_img_gene = layers.concatenate([encoder_img_gene, encoder_gene_img])

        # latent distribution layer
        z_mean = layers.Dense(self.latent_dim, name="z_mean")(encoder_img_gene)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(encoder_img_gene)
        z = Sampling()([z_mean, z_log_var])

        c = layers.concatenate([clinical_inputs, z_mean])
        c = layers.BatchNormalization()(c)
        c = layers.Dense(self.latent_dim, activation="tanh", kernel_initializer=initializers.GlorotUniform(
            seed=None), kernel_regularizer=l2(l2_regularizer))(c)
        c = layers.Dropout(drop)(c)
        c = layers.Dense(n1_unit, activation="tanh", kernel_initializer=initializers.GlorotUniform(
            seed=None), kernel_regularizer=l2(l2_regularizer))(c)
        cox_outputs = layers.Dense(1, use_bias=False, kernel_regularizer=l2(
            l2_regularizer), name='cox_hazard')(c)
        encoder = keras.Model([encoder_img_inputs, encoder_gene_inputs, clinical_inputs], [
                              z_mean, z_log_var, z, cox_outputs], name="encoder")
        encoder.summary()

        """
        Decoder
        """
        # layer_4
        latent_inputs = keras.Input(
            shape=(self.latent_dim,), name='decoder_input')
        decoder_img = layers.Dense(
            56 * 56 * 64, activation="relu", name="decoder_img_layer_4")(latent_inputs)
        decoder_img = layers.Reshape((56, 56, 64))(decoder_img)

        # layer_3
        decoder_img = layers.Conv2DTranspose(
            128, 3, activation="relu", strides=2, padding="same", name="decoder_img_layer_3")(decoder_img)

        # layer_2
        decoder_img = layers.Conv2DTranspose(
            64, 3, activation="relu", strides=2, padding="same", name="decoder_img_layer_2")(decoder_img)
        decoder_gene = layers.Dense(
            self.pathway_shape, activation="relu", name="decoder_pathway_layer")(latent_inputs)

        # layer_1
        decoder_img_outputs = layers.Conv2DTranspose(
            5, 3, activation="sigmoid", padding="same", name="decoder_img_layer_1")(decoder_img)
        decoder_gene_output = layers.Dense(
            self.original_gene_dim, activation="sigmoid", name="decoder_gene_layer_1")(decoder_gene)

        decoder = keras.Model(
            latent_inputs, [decoder_img_outputs, decoder_gene_output], name="decoder")
        # decoder.summary()

        vae = VAE(hp, encoder, decoder, self.pathway_mask)
        adam = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        vae.compile(optimizer=adam)

        return vae

    def evaluate_model(self, data, saved_encoder):
        img_data, gene_data, clinical, y_event, y_time = data
        z_mean, z_log_var, z, pred_risk = saved_encoder.predict(
            [img_data, gene_data, clinical])
        cidx = cindex(list(map(bool, y_event.reshape(-1))),
                      y_time.reshape(-1), pred_risk.reshape(-1))
        return cidx


class VAE(tf.keras.Model):
    def __init__(self, hp, encoder, decoder, pathway_mask, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.pathway_mask = pathway_mask
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.img_reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="img_reconstruction_loss"
        )
        self.gene_reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="gene_reconstruction_loss"
        )
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        self.cox_loss_tracker = tf.keras.metrics.Mean(name="cox_loss")
        self.C_max = tf.constant(
            hp.Choice("c_stop", [30, 35]), dtype=tf.float32)
        # self.C_max = tf.constant([max_capacity], dtype=tf.float32)
        self.C_stop_iter = Capacity_max_iter
        self.num_iter = 0

        self.k_cl = hp.Float("k_cl", min_value=0.6, max_value=0.8)
        self.k_vl = 1 - self.k_cl

        self.loss_fn = CoxPHLoss()
        self.val_cindex_metric = CindexMetric()

    @property
    def metrics(self):
        return [
            self.val_cindex_metric,
            self.cox_loss_tracker
        ]

    def update_weights(self, layer):
        old_weight = layer.get_weights()
        new_weight = layer.get_weights()[0]*self.pathway_mask
        updated_weight = old_weight
        updated_weight[0] = new_weight
        layer.set_weights(updated_weight)

    # train steps

    def train_step(self, data):
        self.num_iter += 1
        regularisation_loss = 0
        with tf.GradientTape() as tape:

            img_data, gene_data, clinical = data[0]
            y_event, y_time = data[1]
            z_mean, z_log_var, z, pred_risk = self.encoder(
                [img_data, gene_data, clinical])
            reconstruction = self.decoder(z)
            image_reconstruction = reconstruction[0]
            gene_reconstruction = reconstruction[1]

            kl_loss = -0.5 * (1 + z_log_var -
                              tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

            img_reconstruction_loss_ = tf.keras.metrics.mean_squared_error(
                img_data, image_reconstruction)
            img_reconstruction_loss = tf.reduce_mean(
                img_reconstruction_loss_) * 224*224
            gene_reconstruction_loss = tf.keras.metrics.mean_squared_error(
                gene_data, gene_reconstruction)

            if len(self.encoder.losses) > 0:
                regularisation_loss = tf.math.add_n(self.encoder.losses)

            cox_loss = self.loss_fn(y_true=[y_event, y_time], y_pred=pred_risk)
            # https://arxiv.org/pdf/1804.03599.pdf
            C = tf.clip_by_value(
                self.C_max/self.C_stop_iter * self.num_iter, 0, self.C_max.numpy())
            img_total_loss = K.mean(
                img_reconstruction_loss + gamma * kld_weight * K.abs(kl_loss - C))
            gene_total_loss = K.mean(
                gene_reconstruction_loss + gamma * kld_weight * K.abs(kl_loss - C))
            total_loss = K.mean(img_total_loss + gene_total_loss) * self.k_vl + \
                (cox_loss + regularisation_loss + kl_loss) * self.k_cl

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.update_weights(self.encoder.get_layer('Pathway_Layer'))
        self.total_loss_tracker.update_state(total_loss)
        self.img_reconstruction_loss_tracker.update_state(
            img_reconstruction_loss)
        self.gene_reconstruction_loss_tracker.update_state(
            gene_reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.cox_loss_tracker.update_state(cox_loss)
        self.val_cindex_metric.update_state(y_time, y_event, pred_risk)
        return {
            "loss": self.total_loss_tracker.result(),
            "img_reconstruction_loss": self.img_reconstruction_loss_tracker.result(),
            "gene_reconstruction_loss": self.gene_reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "cox_loss": self.cox_loss_tracker.result(),
            "cindex": self.val_cindex_metric.result(),
        }

    # test steps
    def test_step(self, data):
        img_data, gene_data, clinical = data[0]
        y_event, y_time = data[1]
        regularisation_loss = 0
        z_mean, z_log_var, z, pred_risk = self.encoder(
            [img_data, gene_data, clinical], training=False)
        reconstruction = self.decoder(z, training=False)
        image_reconstruction = reconstruction[0]
        gene_reconstruction = reconstruction[1]

        kl_loss = -0.5 * (1 + z_log_var -
                          tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

        img_reconstruction_loss_ = tf.keras.metrics.mean_squared_error(
            img_data, image_reconstruction)
        img_reconstruction_loss = tf.reduce_mean(
            img_reconstruction_loss_) * 224*224
        gene_reconstruction_loss = tf.keras.metrics.mean_squared_error(
            gene_data, gene_reconstruction)

        if len(self.encoder.losses) > 0:
            regularisation_loss = tf.math.add_n(self.encoder.losses)

        cox_loss = self.loss_fn(y_true=[y_event, y_time], y_pred=pred_risk)
        # https://arxiv.org/pdf/1804.03599.pdf
        C = tf.clip_by_value(self.C_max/self.C_stop_iter *
                             self.num_iter, 0, self.C_max.numpy())
        img_total_loss = K.mean(
            img_reconstruction_loss + gamma * kld_weight * K.abs(kl_loss - C))
        gene_total_loss = K.mean(
            gene_reconstruction_loss + gamma * kld_weight * K.abs(kl_loss - C))
        total_loss = K.mean(img_total_loss + gene_total_loss) * self.k_vl + \
            (cox_loss + regularisation_loss + kl_loss) * self.k_cl

        self.total_loss_tracker.update_state(total_loss)
        self.img_reconstruction_loss_tracker.update_state(
            img_reconstruction_loss)
        self.gene_reconstruction_loss_tracker.update_state(
            gene_reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.cox_loss_tracker.update_state(cox_loss)
        self.val_cindex_metric.update_state(y_time, y_event, pred_risk)
        return {
            "loss": self.total_loss_tracker.result(),
            "img_reconstruction_loss": self.img_reconstruction_loss_tracker.result(),
            "gene_reconstruction_loss": self.gene_reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "cox_loss": self.cox_loss_tracker.result(),
            "cindex": self.val_cindex_metric.result()
        }
