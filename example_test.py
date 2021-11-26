# Nov 25th 2021 written by Garam Lee

# This code provides interfaces to test the pre-trained model. Note that training phase is not included in this code
# Classification is done via two models: GRU and Random forest.
# GRU is used to convert intra-operative features (3-order tensors) into 2-order feature representations.
# Then the representation is concatenated with pre-operative features to generate the new represetation that contains both pre-operative and intra-operative information.
# Those integrated representation is used to run Random forest for the final prediction (massive transfusion)

import pandas as pd
import numpy as np
import pickle

from sklearn import metrics
import keras

import config
import util
import batch_generator

# Load data and models #######################################
intraop_dir = 'data/intraop_data/'
preop_filename = 'data/example_test_data.csv'

pretrain_intra_filename = 'pretrained_model/ISSHctv10'
pretrain_final_filename = 'pretrained_model/ISSHctRF10'

test_data = pd.read_csv(preop_filename,sep=',')
pretrain_intra_model = keras.models.load_model(pretrain_intra_filename)
pretrain_final_model = pickle.load(open(pretrain_final_filename,'rb'))
hidden_layer_generator = keras.Model(inputs=pretrain_intra_model.input, outputs=pretrain_intra_model.layers[3].output)

intra_X = test_data['filename']
preop_X = test_data[config.preop_features]
y = test_data['masstf']
###############################################################

bg = batch_generator.CustomGenerator(intraop_dir, intra_X, None, config.intra_features, len(intra_X)) 
intra_representation = hidden_layer_generator.predict(bg)[:,-1,:] # hidden representation at the last time point resulted from GRU
integrated_representation = np.concatenate((intra_representation, preop_X), axis=1) # data integration

y_pred = pretrain_final_model.predict_proba(integrated_representation)[:,1]

fpr, tpr, _ = metrics.roc_curve(y, y_pred)
cutoff = 0.02                   # cutoff is determined based on development dataset
y_pred_class = [1 if x>=cutoff else 0 for x in  y_pred]
cm = metrics.confusion_matrix(y, y_pred_class)
tn, fn, tp, fp = cm[0][0], cm[1][0], cm[1][1], cm[0][1]

sensitivity = tp/(tp+fn)
specificity = tn/(fp+tn)
PPV = tp/(tp+fp)
NPV = tn/(tn+fn)
    
print("auROC:",metrics.auc(fpr, tpr))
print("auPRC:", metrics.average_precision_score(y, y_pred))
print("sensitivity:", sensitivity)
print("specificity:", specificity)
print("PPV:", PPV)
print("NPV:", NPV)
