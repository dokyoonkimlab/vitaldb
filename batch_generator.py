import pandas as pd
import numpy as np
import os


import keras
from keras.preprocessing import sequence
from keras.utils import to_categorical

maxlen = 1200

class CustomGenerator(keras.utils.Sequence):
    def __init__(self, filename_prefix, filenames, labels, variable_selector, batch_size):
        self.filename_prefix=filename_prefix
        self.filenames =filenames
        self.labels = labels
        self.batch_size = batch_size
        self.variable_selector = variable_selector

    def __len__(self):
        return np.ceil(len(self.filenames) / self.batch_size).astype(np.int)

    def __getitem__(self, idx):
        batch_x = self.filenames[idx*self.batch_size : (idx+1) * self.batch_size]
        
        X = [pd.read_csv(self.filename_prefix + filename, sep=',')[self.variable_selector].values for filename in batch_x]
        X = sequence.pad_sequences(X, maxlen=maxlen, padding='pre', truncating='pre', dtype='float32')
        if not self.labels is None:
            batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]
            y = to_categorical(batch_y)
        else:
            y = None
            
        return X, y
    
