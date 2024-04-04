# -*- coding: utf-8 -*-
"""
@author: SURAJ
"""
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.regularizers import l2
from utils.SurvivalUtils import CoxPHLoss, CindexMetric, cindex
import keras_tuner as kt
from tensorflow.keras import initializers

tf.config.run_functions_eagerly(True)

learning_rate = 1e-3
decay = 0.1

# scheduled learning rate
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=learning_rate, decay_steps=100, decay_rate=decay)


class Clinical_Cox_model(kt.HyperModel):

    def __init__(self, clinical_dim):

        self.clinical_dim = clinical_dim

    def build_model(self, hp):

        l2_regularizer = hp.Float(
            "l2_regularisation", min_value=1e-3, max_value=1e-1, sampling="log")
        drop = 0.9  # hp.Choice('dropout',[0.8, 0.9])
        n1_unit = hp.Choice('n1', [2, 4], default=2)

        clinical_inputs = keras.Input(
            shape=(self.clinical_dim,), name='clinical_input')
        c = layers.BatchNormalization()(clinical_inputs)
        c = layers.Dense(self.clinical_dim, activation="tanh", kernel_initializer=initializers.GlorotUniform(
            seed=None), kernel_regularizer=l2(l2_regularizer))(c)
        c = layers.Dropout(drop)(c)
        c = layers.Dense(n1_unit, activation="tanh", kernel_initializer=initializers.GlorotUniform(
            seed=None), kernel_regularizer=l2(l2_regularizer))(c)
        cox_outputs = layers.Dense(1, use_bias=False, kernel_regularizer=l2(
            l2_regularizer), name='cox_hazard')(c)
        clinical_model = keras.Model(clinical_inputs, cox_outputs, name="Cliniical cox model")
        clinical_model.summary()

        cModel = CoxModel(hp, clinical_model)
        adam = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        cModel.compile(optimizer=adam)

        return cModel

    def evaluate_model(self, data, saved_encoder):
        clinical, y_event, y_time = data
        pred_risk = saved_encoder.predict(clinical)
        cidx = cindex(list(map(bool, y_event.reshape(-1))),
                      y_time.reshape(-1), pred_risk.reshape(-1))
        return cidx


class CoxModel(tf.keras.Model):
    def __init__(self, hp, clinical_model, **kwargs):
        super(CoxModel, self).__init__(**kwargs)
        self.clinical_model = clinical_model
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.cox_loss_tracker = tf.keras.metrics.Mean(name="cox_loss")

        self.loss_fn = CoxPHLoss()
        self.val_cindex_metric = CindexMetric()

    @property
    def metrics(self):
        return [
            self.val_cindex_metric,
            self.cox_loss_tracker
        ]

    # train steps

    def train_step(self, data):
        regularisation_loss = 0
        with tf.GradientTape() as tape:

            clinical = data[0]
            y_event, y_time = data[1]
            pred_risk = self.clinical_model(clinical)

            if len(self.clinical_model.losses) > 0:
                regularisation_loss = tf.math.add_n(self.clinical_model.losses)

            cox_loss = self.loss_fn(y_true=[y_event, y_time], y_pred=pred_risk)
            total_loss = (cox_loss + regularisation_loss)

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.cox_loss_tracker.update_state(cox_loss)
        self.val_cindex_metric.update_state(y_time, y_event, pred_risk)
        return {
            "loss": self.total_loss_tracker.result(),
            "cox_loss": self.cox_loss_tracker.result(),
            "cindex": self.val_cindex_metric.result(),
        }

    # test steps
    def test_step(self, data):
        clinical = data[0]
        y_event, y_time = data[1]
        regularisation_loss = 0
        pred_risk = self.clinical_model(clinical, training=False)

        if len(self.clinical_model.losses) > 0:
            regularisation_loss = tf.math.add_n(self.clinical_model.losses)

        cox_loss = self.loss_fn(y_true=[y_event, y_time], y_pred=pred_risk)

        total_loss = (cox_loss + regularisation_loss)

        self.total_loss_tracker.update_state(total_loss)
        self.cox_loss_tracker.update_state(cox_loss)
        self.val_cindex_metric.update_state(y_time, y_event, pred_risk)
        return {
            "loss": self.total_loss_tracker.result(),
            "cox_loss": self.cox_loss_tracker.result(),
            "cindex": self.val_cindex_metric.result()
        }
