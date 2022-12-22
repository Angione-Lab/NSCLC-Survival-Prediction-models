import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
#from sklearn_pandas import DataFrameMapper
from baselineModels.pycox.Model import CoxModel

import torch
import torchtuples as tt

from pycox.datasets import metabric
from pycox.models import CoxPH, CoxCC, LogisticHazard
from pycox.evaluation import EvalSurv
from utils.DataLoader import load_data  
import pandas as pd

SEED = 20
np.random.seed(SEED)
_ = torch.manual_seed(SEED)


feature_dataset, clinical_features, y_event, y_survival_time = load_data('lat_image_gene')

in_features = feature_dataset.shape[1]
clic_features = clinical_features.shape[1]
out_features = 1
batch_norm = True
dropout = 0.1
output_bias = False


batch_size = 64
epochs = 100 
outer_fold = 5

# evaluate a model using k-fold cross-validation
def nested_cross_validate_model(radio_genomics_df,clinical_features,y_event,y_time):
    c_index, best_hyperparameters, shap_values, pred_risk_list,high_risk_features_list, high_risk_clinical_list, time, event = list(),list(), list(), list(), list(), list(), list(), list()
        
    # prepare cross validation
    k_outer_fold = StratifiedKFold(outer_fold, shuffle=True, random_state=SEED)
    # enumerate splits
    i = 1
    #outer fold
    for train_test_ix, validation_idx  in k_outer_fold.split(radio_genomics_df, y_event):
        
        train_test_rad = radio_genomics_df.loc[train_test_ix].reset_index(drop=True)
        validation_rad = radio_genomics_df.loc[validation_idx].reset_index(drop=True) 
        train_test_clic = clinical_features.loc[train_test_ix].reset_index(drop=True)
        validation_clic = clinical_features.loc[validation_idx].reset_index(drop=True) 
        train_test_event = y_event.loc[train_test_ix].reset_index(drop=True)
        validation_test_event = y_event.loc[validation_idx].reset_index(drop=True)
        train_test_time = y_time.loc[train_test_ix].reset_index(drop=True)
        validation_time = y_time.loc[validation_idx].reset_index(drop=True)
        
        train_test_rad = StandardScaler().fit_transform(train_test_rad)
        validation_rad = StandardScaler().fit_transform(validation_rad)
        train_test_clic = MinMaxScaler().fit_transform(train_test_clic)
        validation_clic = MinMaxScaler().fit_transform(validation_clic)
        
        
        train_rad, test_rad, train_clic, test_clic, train_event, test_event, train_time, test_time = \
            train_test_split(train_test_rad,train_test_clic,train_test_event,train_test_time,  test_size=0.15, \
                             stratify=train_test_event, shuffle = True, random_state=SEED )
        
        x_train = [np.array(train_rad).astype('float32') , np.array(train_clic).astype('float32')]
        x_test =  [np.array(test_rad).astype('float32') , np.array(test_clic).astype('float32')]
        x_val =  [np.array(validation_rad).astype('float32') , np.array(validation_clic).astype('float32')]
        
        y_train =(train_time['survival_time'].values, train_event['Survival Status'].values)
        y_test = (test_time['survival_time'].values, test_event['Survival Status'].values)
   
        net = CoxModel(in_features, clic_features, 128, 64, 32)
        model = CoxPH(net, tt.optim.Adam)
               
        lrfinder = model.lr_finder(x_train, y_train,  tolerance=50)
        _ = lrfinder.plot()
        
        bestlr = lrfinder.get_best_lr()
        model.optimizer.set_lr(bestlr)
        
        callbacks = [tt.callbacks.EarlyStopping()]
        verbose = True
        
        log = model.fit(x_train, y_train, batch_size, epochs, callbacks, verbose,
                        val_data=(x_test, y_test), val_batch_size=batch_size)
        
        _ = log.plot()
        
        test = x_test, y_test
        model.partial_log_likelihood(*test).mean()
        
        _ = model.compute_baseline_hazards()
        
        
        #Predictions
        
        surv = model.predict_surv_df(x_val)
        #print(surv)
        
        surv.iloc[:, :5].plot()
        plt.ylabel('S(t | x)')
        _ = plt.xlabel('Time')
        plt.show()
        ev = EvalSurv(surv, validation_time['survival_time'].values, validation_test_event['Survival Status'].values, censor_surv='km')
        c_index.append(ev.concordance_td())
     
        
        i = i+1
    return c_index, best_hyperparameters, pred_risk_list, shap_values, time, event, high_risk_features_list, high_risk_clinical_list


c_index, best_hyperparameters, pred_risk_list, shap_values, time, event, high_risk_features_list, high_risk_clinical_list = nested_cross_validate_model(pd.DataFrame(np.array(feature_dataset)),clinical_features,y_event,y_survival_time)

print('mean c_index', np.mean(c_index)) 
print('std c_index', np.std(c_index)) 
print(c_index)
#a = np.array(pred_risk_list).reshape(130, 1)