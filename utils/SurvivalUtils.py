# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 11:57:37 2021

@author: suraj
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Optional
from sksurv.metrics import concordance_index_censored
from tensorflow.keras.metrics import Metric
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
from lifelines.statistics import logrank_test
from lifelines.plotting import add_at_risk_counts
import shap
from keras.callbacks import Callback

class CustomCheckpoint(Callback):

    def __init__(self, filepath, model,  monitor, save_best_only=True, mode='max', patience=10):
        self.monitor = monitor
        print('\n checkpoint initialised: ', filepath)
        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = 0
 
        self.filepath = filepath
        self.model = model
        self.patience = patience
    
    def on_epoch_end(self, epoch, logs=None):
       
        current = logs.get(self.monitor)
        if self.monitor_op(current, self.best) and np.greater(epoch , self.patience):
            print('\n saving bese model with {} , {} \n'.format(self.monitor,current ))
            self.best = current
            # self.encoder.save_weights(self.filepath, overwrite=True)
            self.model.encoder.save(self.filepath + '/encoder', overwrite=True) 
            self.model.decoder.save(self.filepath + '/decoder', overwrite=True) 


def h_vae_shap_values(model, radgeno_data, clinical_data):
    
    gradModel = tf.keras.Model(
        inputs=[model.encoder.input,],
        outputs=[model.encoder.layers[-1].output])
    
    exp = shap.GradientExplainer(gradModel,[radgeno_data, clinical_data])
    shap_values, shap_values_var = exp.shap_values([radgeno_data, clinical_data],return_variances=True)
    
    return shap_values, shap_values_var

def x_vae_shap_values(model, img, rna, clinical_data):
    
    gradModel = tf.keras.Model(
        inputs=[model.encoder.input,],
        outputs=[model.encoder.layers[-1].output])
    
    exp = shap.GradientExplainer(gradModel,[img, rna, clinical_data])
    shap_values, shap_values_var = exp.shap_values([img, rna, clinical_data],return_variances=True)
    
    return shap_values, shap_values_var



def R_set(x):
	n_sample = x.shape[0]
	matrix_ones = tf.ones([n_sample, n_sample], tf.int32)
	indicator_matrix = tf.compat.v1.matrix_band_part(matrix_ones, -1, 0)

	return(indicator_matrix)

def safe_normalize(x):

    x_min = tf.reduce_min(x, axis=0)
    c = tf.zeros_like(x_min)
    norm = tf.where(x_min < 0, -x_min, c)
    return x + norm


def logsumexp_masked(risk_scores,
                     mask,
                     axis: int = 0,
                     keepdims: Optional[bool] = None):
    """Compute logsumexp across `axis` for entries where `mask` is true."""
    risk_scores.shape.assert_same_rank(mask.shape)

    with tf.name_scope("logsumexp_masked"):
        mask_f = tf.cast(mask, risk_scores.dtype)
        risk_scores_masked = tf.math.multiply(risk_scores, mask_f)
        amax = tf.reduce_max(risk_scores_masked, axis=axis, keepdims=True)
        risk_scores_shift = risk_scores_masked - amax
        exp_masked = tf.math.multiply(tf.exp(risk_scores_shift), mask_f)
        exp_sum = tf.reduce_sum(exp_masked, axis=axis, keepdims=True)
        output = amax + tf.math.log(exp_sum)
        if not keepdims:
            output = tf.squeeze(output, axis=axis)
    return output

def cox_log_likelihood(event, time, risk):

    # sort data
    t_ordered, idx = tf.nn.top_k(time[:,0], k=time.shape[0])
    e_ordered = tf.gather(event, indices=idx, axis=0)
    risk_ordered = tf.gather(risk, indices=idx, axis=0)
    e_ordered = tf.cast(e_ordered, risk_ordered.dtype)

    # compute likelihood
    sum_risk = tf.reduce_sum(e_ordered * risk_ordered)
    log_sums = tf.math.log(tf.cumsum(tf.exp(risk_ordered)))
    log_sums = tf.reduce_sum(e_ordered * log_sums)

    lcpl = sum_risk - log_sums
    return lcpl



class CoxPHLoss(tf.keras.losses.Loss):
    """Negative partial log-likelihood of Cox's proportional hazards model."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)            

    def call(self,  y_true, y_pred):

        event, time = y_true
        predictions = y_pred
        
        # sort data
        time, idx = tf.nn.top_k(time[:,0], k=time.shape[0])
        event = tf.gather(event, indices=idx, axis=0)
        predictions = tf.gather(predictions, indices=idx, axis=0)
        riskset = R_set(time) 

        event = tf.cast(event, predictions.dtype)
        predictions = safe_normalize(predictions)

        pred_t = tf.transpose(predictions)

        rr = logsumexp_masked(pred_t, riskset, axis=1, keepdims=True)
        assert rr.shape.as_list() == predictions.shape.as_list()

        losses = tf.math.multiply(event, rr - predictions)
       
        return losses

def cindex(event, time, prediction):
    """
    

    Parameters
    ----------
    event : TYPE
        DESCRIPTION.event
    time : TYPE
        DESCRIPTION.time
    prediction : TYPE
        DESCRIPTION. predicted risk

    Returns
    -------
    TYPE
        DESCRIPTION. Cindex

    """
    cidx = concordance_index_censored(event, time, prediction)
    return cidx[0]

class CindexMetric(Metric):
    """Computes concordance index across one epoch."""

    def reset_state(self):
        """Clear the buffer of collected values."""
        self._data = {
            "label_time": [],
            "label_event": [],
            "prediction": []
        }

    def update_state(self, y_time, y_event, y_pred):

        self._data["label_time"].append(tf.squeeze(y_time).numpy())
        self._data["label_event"].append(tf.squeeze(y_event).numpy())
        self._data["prediction"].append(tf.squeeze(y_pred).numpy())
        #print(y_event)

    def result(self):
        results = 0
        try:
            data = {}
            for k, v in self._data.items():
                data[k] = np.concatenate(v)

            results = cindex(
                data["label_event"] == 1,
                data["label_time"],
                data["prediction"])
        except Exception as ex:
            results = 0
            print(ex)
        return results



# KM Plot
def km_plot(y_event, y_time, pred_risk):
    #clinical_features[['Time to Death (day, pred_risk):
    median_risk = np.median(pred_risk)
    train_data = pd.concat([y_event, y_time], axis=1)
    #clinical_features[['Time to Death (days)', 'Survival Status']]
    train_data['Hazard'] = pred_risk
    
    group1 = train_data[train_data['Hazard'] <= median_risk]
    group2 = train_data[train_data['Hazard'] > median_risk]
    #get count of patients and plot in figure.
    
    
    T=group1['survival_time']
    E=group1['Survival Status']
    T1=group2['survival_time']
    E1=group2['Survival Status']
    
    results = logrank_test(T, T1, event_observed_A=E, event_observed_B=E1)
    pvalue_pred = results.p_value
    
    
    fig = plt.figure(figsize=(7,5))
    ax = plt.subplot(111)
    f1 = KaplanMeierFitter()
    f1.fit_right_censoring(T, E, label="Low Risk")
    f1.plot(ax=ax)
    f2 = KaplanMeierFitter()
    f2.fit_right_censoring(T1, E1, label="High Risk")
    f2.plot(ax=ax)
    ax.text(10,0.5, 'p-val: {0:.8f}'.format(pvalue_pred), fontsize=10)
    ax.set_xlabel('Time in days')
    ax.set_ylabel('Survival Probability')
    add_at_risk_counts(f1, f2, ax=ax)
    plt.tight_layout()
    #ax.grid(True)
    plt.title('Survival Analysis')
    plt.show()
    #plt.savefig('Figures/RadioGenomic_Survival_Analysis.png')