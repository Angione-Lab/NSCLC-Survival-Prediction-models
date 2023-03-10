# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Top Level interface with helper functions to run baselines and experiments.

This is the top level module that contains two functions, baseline_experiment()
and experiment() that run the survival analysis baselines and methods on the
SEER and SUPPORT datasets in a cross validation fashion. The module then outputs
the Expected Calibration Error and ROC characteristic at various event time
quantiles.

  Typical usage example:
  from dcm import deep_cox_mixtures
  deep_cox_mixtures.experiment(data='SUPPORT')
  deep_cox_mixtures.baseline_experiment(data='SUPPORT')

"""

import dill as pkl

import baselineModels.DeepCoxMixture.dcm.baseline_models as baseline_models
import baselineModels.DeepCoxMixture.dcm.data_utils as data_utils
import baselineModels.DeepCoxMixture.dcm.models as models
import baselineModels.DeepCoxMixture.dcm.plots as plots
from utils.DataLoader import load_data  
import numpy as np
import pandas as pd

import os
import random
import logging

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from lifelines import KaplanMeierFitter


def store_model(dataset, model, trained_model, params):
  
  path = 'models/'+dataset+'/'+model+'/'
  try:
    os.makedirs(path)
  except Exception as e:
    pass

  if model == 'dcm':
        
    logging.info("Saving DCM")
    existing = os.listdir(path)
    count = max([0]+[int(f[:-4]) for f in existing])+1
    trained_model.save(path+str(count)+'.pkl')

  elif model == 'rsf':
        
    logging.info("Saving RSF")
        
    existing = os.listdir(path)
    count = max([0]+[int(f) for f in existing])+1
    
    path += str(count)+'/'
    try:
      os.makedirs(path)
    except Exception as e:
      pass
        
    for fold in trained_model:
      trained_model[fold].save(path+'/'+str(fold))
      f = open(path+'/'+'params.dict', 'wb')
      pkl.dump(params,f)
      f.flush()
      f.close()
           
  else:    
    
    existing = os.listdir(path)
    count = max([0]+[int(f[:-4]) for f in existing])+1
    
    f = open(path+str(count)+'.pkl', 'wb')
    pkl.dump((trained_model, params), f)
    f.flush()
    f.close()

  return 

def load_dataset(dataset, cv_folds, prot_att, fair_strategy, quantiles, dim_red=False):
    
  if dataset == 'NSCLC':
      
    feature_dataset, clinical_features, y_event, y_survival_time = load_data('lat_image_gene')
    conc_features = np.concatenate([feature_dataset.values, clinical_features.values], axis=1)
    
    t = y_survival_time.values.reshape(-1)
    e = y_event.values.reshape(-1)
    x = conc_features
    a = clinical_features['Age at Histological Diagnosis'].values
    
        
  else:
    raise NotImplementedError("Dataset:", dataset, " not implemented.")
    
  folds = np.array((list(range(cv_folds)) * (len(a) // cv_folds + 1))[:len(a)])

  quantiles = np.quantile(t[e == 1], quantiles)

  print(quantiles)
  
  if dim_red:
      logging.info("Running PCA, Please Stand by...")
      x = PCA(n_components=dim_red).fit_transform(x)
      print (x.shape)
  return (x, t, e, a), folds, quantiles
 
def baseline_experiment(dataset='SUPPORT', quantiles=(0.25, 0.5, 0.75),
                        prot_att='race', groups=('other', 'white'),
                        model='cph', dim_red=False, fair_strategy=None, 
                        cv_folds=5, seed=100,
                        hyperparams=None, plot=True, store=False, adj='KM'):

  """Top level interface to train and evaluate a baseline survival model.

  This is the top level function that is designed to be called directly from
  inside a jupyter notebook kernel. This function allows the user to run
  several baselines survival analysis models on the SEER, SUPPORT & FLCHAIN
  datasets in a cross validation fashion. The function then plots and outputs
  the Expected Calibration Error and ROC characteristic at various event time
  quantiles.

  Parameters
  ----------
  dataset: str
      a string that determines the dataset to run experiments on. 
      One of "FLCHAIN" or "SUPPORT".
  quantiles: list
      a list of event time quantiles at which the models are to be evaluated.
  prot_att: str
      a string that specifies the column in the dataset that is to be treated
      as a protected attribute.
  groups: list
      a list of two groups on which the survival analysis models are to be
      evaluated vis a vis accuracy and fairness.
  model: str
      the choice of the baseline survival analysis model. One of "cph", "dcph",
      "dsm", "aft", "rsf"
  fair_strategy: str
      string that specifies the fairness strategy to be used while running the
      experiment. One of None, "unawareness", "coupled".
  cv_folds: int
      int that determines the number of Cross Validation folds.

  Returns:
      a Matplotlib figure with the ROC Curves and Reliability (Calibration) curves
      at various event quantiles.

  """
  try:
    np.random.seed(seed)
    
    (x, t, e, a), folds, quantiles = load_dataset(dataset, 
                                                  cv_folds, 
                                                  prot_att, 
                                                  fair_strategy,
                                                  quantiles, 
                                                  dim_red )

    if fair_strategy == 'coupled':
      trained_model = {}
      for grp in groups:
        trained_model[grp] = baseline_models.train_model(x[a == grp],
                                                         t[a == grp],
                                                         e[a == grp],
                                                         folds=folds[a == grp],
                                                         model=model,
                                                         params=hyperparams,
                                                         random_state=seed)
    else:
      trained_model = baseline_models.train_model(x, t, e, folds, model=model, params=hyperparams, 
                                                  random_state=seed)
      logging.info("All Folds Trained...")
    if store:
      logging.info("Storing Models...")
      store_model(dataset, model, trained_model, hyperparams)
  
    outputs = predict(trained_model, model, x, t, e, a, folds, quantiles, fair_strategy)

    if plot:
      results = plots.plot_results(outputs, x, e, t, a, folds,
                                   groups, quantiles, strat='quantile', adj=adj)
      return results

    else:
      return outputs

  except Exception as e:
    print(e)
    return None
  

def experiment(dataset='SUPPORT', quantiles=(0.25, 0.5, 0.75), prot_att='race',
               groups=('other', 'white'), model='dcm', adj='KM',
               cv_folds=5, seed=100, hyperparams=None, plot=True, store=False):

  """Top level interface to train and evaluate proposed survival models.

  This is the top level function that is designed to be called directly from
  inside a jupyter notebook kernel. This function allows the user to run
  one of the proposed survival analysis models on the SUPPORT datasets
  in a cross validation fashion. The function then plots and
  outputs the Expected Calibration Error and ROC characteristic at various
  event time quantiles.

 
  Parameters
  ----------
  dataset: str
      a string that determines the dataset to run experiments on. 
      One of "FLCHAIN" or "SUPPORT".
  quantiles: list
      a list of event time quantiles at which the models are to be evaluated.
  prot_att: str
      a string that specifies the column in the dataset that is to be treated
      as a protected attribute.
  groups: list
      a list of strings indicating groups on which the survival analysis 
      models are to be evaluated vis a vis discrimination and calibration.
  model: str
      the choice of the proposed survival analysis model. 
      currently supports only "dcm".
  adj: str
      the choice of adjustment for the L1-ECE: one of 
      * 'IPCW': Inverse Propensity of Censoring Weighting.
      * 'KM': Kaplan-Meier.
  cv_folds: int
      int that determines the number of Cross Validation folds.
  seed: int
      numpy random seed.
  hyperparams: dict
      a dict with hyperparams for the DCM model.
  plot: bool
      binary flag to determine if the results are to be plotted.
  store: bool
      whether the models/results are to be stored to disk.
  
  Returns:
      a Matplotlib figure with the ROC Curves and Reliability (Calibration) curves
      at various event quantiles.

  """
  
  try:
    
    random.seed(seed)
    np.random.seed(seed)

    fair_strategy = None

    (x, t, e, a), folds, quantiles = load_dataset(dataset, cv_folds, prot_att, fair_strategy, quantiles)

    trained_model = models.train_model(x, t, e, a, folds, groups, params=hyperparams, random_state=seed)

    if store:
      store_model(dataset, model, trained_model, hyperparams)

    outputs = predict(trained_model, model, x, t, e, a, folds, quantiles, fair_strategy)

    if plot:
      results = plots.plot_results(outputs, x, e, t, a, folds, groups,
                           quantiles, strat='quantile', adj=adj)
      return results

    else:
      return outputs
    
  except Exception as e:
        
    print(e)
    return None
        


def predict(trained_model, model, x, t, e, a, folds, quantiles, fair_strategy):
    
  outputs = {}

  for quant in quantiles:
    if model in ['dcm']: 
      outputs[quant] = models.predict_scores(trained_model, None, x, a, folds,
                                            quant)
    else:
      outputs[quant] = baseline_models.predict_scores(trained_model, model,
                                                     fair_strategy, x, a, folds,
                                                     quant)
  return outputs

def display_results(results):
  """Helper function to pretty print the results from experiment. 
  
  Args:
    results: output of dcm.deep_cox_mixtures.experiment or 
             dcm.deep_cox_mixtures.baseline_experiment   
  
  """
    
  quantiles = results.keys()
  metrics = ['AuC', 'Ctd', 'BrS', 'ECE']
  
  for quant in quantiles:
    
    print("".join(["-"]*11*4))
    print("Event Horizon:", quant)
    print("".join(["-"]*11*4))
    print("{: <8}".format(""), end="  |")
    
    groups = results[quant][0]
    
    print("{: <30}".format("           Groups"), end="  |")
    print("")
    print("".join(["-"]*11*4))

    print("{: <8}".format("Metric"), end="  |")

    for group in groups:
      print("{: >8}".format(group), end="  |")
    
    print()
    print("".join(["-"]*11*4))
    
    i = 0
    for metric in metrics: 
      print()
      print("{: <8}".format(metric), end="  |")
      
      for group in groups:
        print("{: >8}".format(round(results[quant][i][group], 5)),  end="  |" )

      i+=1
    print()
    print("".join(["-"]*11*4))
    print()
