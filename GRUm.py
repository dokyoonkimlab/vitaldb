# Oct24th 2020 Garam Lee
import keras
from keras import Sequential, Model
from keras.layers import Dense, GRU, Input, Lambda,Dropout, LSTM

max_length = 1200

class Singlem(object):
    def __init__(self, input_dim):
        self.input_dim = input_dim
        input = Input(shape=(max_length, input_dim))

        def m1(input):
            z = GRU(100, activation='tanh', return_sequences=True)(input)
            d1 = Dense(50, activation='relu')
            d2 = Dense(30, activation='relu')
            d3 = Dense(2, activation='softmax')
            
            z = d1(z)
            z = d2(z)
            z = d3(z)
            return z
            
        z = m1(input)
        out = Lambda(lambda x: x[:,-1,:])(z)
        model = Model(input, out)
        latent = Model(input, z)
        model.compile(optimizer='adam', loss='binary_crossentropy')
        self.model = model
        self.latent = latent

    def fit(self, bg, batch_size, epochs):
        self.model.fit(bg, epochs=epochs, batch_size=batch_size, verbose=1)

    def predict(self, X):
        return self.model.predict(X)
        
    def generate_trisk_score(self, X):
        return self.latent.predict(X)
