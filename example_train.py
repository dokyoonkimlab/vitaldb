# Nov 25th 2021 written by Garam Lee in Kimlab

# This code provides interfaces to train models based on synthetic samples.
# Note that pretrained model, which we provide in the directory "pretrained_model" was not trained with the samples provided in this code.

# Model training consists of two phases:
# 1. training GRU to analyze intra-operative features
# 2. training Random Forest



import pandas as pd
import numpy as np
import os
import pickle

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import keras

import config
import util
import batch_generator

import GRUm

# Load data and models #######################################
model_save_dir = 'trained_model/'
intraop_dir = 'data/intraop_data/'
preop_filename = 'data/example_test_data.csv'

data = pd.read_csv(preop_filename,sep=',')
###############################################################


def learning_phase1(X_train, y_train):
    intra_features = config.intra_features
    batch_size = 64
    epochs = 20
    bg = batch_generator.CustomGenerator(intraop_dir, X_train['filename'], y_train, intra_features, batch_size)
    gru = GRUm.Singlem(len(config.intra_features))
    gru.fit(bg, batch_size, epochs)
    return gru, bg

def learning_phase2(X_train, y_train):
    clf = RandomForestClassifier(n_estimators=150, max_depth=7)
    clf.fit(X_train, y_train)
    return clf

def save_models(gru, clf):
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
        
    gru.model.save(model_save_dir + 'GRU')
    pickle.dump(clf, open(model_save_dir + 'RF', 'wb'))
    

def main():
    X = data
    y = data['masstf'].values
    
    gru, bg = learning_phase1(X,y)
    hidden_layer_generator = keras.Model(inputs=gru.model.input, outputs=gru.model.layers[3].output)
    intra_representation = hidden_layer_generator.predict(bg)[:,-1,:] # hidden representation at the last tie point resulted from GRU
    integrated_representation = np.concatenate((intra_representation, X[config.preop_features]), axis=1)
    clf = learning_phase2(integrated_representation, y)
    save_models(gru, clf)
    
if __name__ == '__main__':
    main()
