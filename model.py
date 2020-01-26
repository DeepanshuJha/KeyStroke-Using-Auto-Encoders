import keras
from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import TensorBoard
import numpy as np 
import tensorflow
import os

class Models:

    def __init__(self, h, w):
        self.h = h
        self.w = w
        
    def _encoder(self):
        inputs = Input(shape=(self.w, ))
        encoded = Dense(6, activation = 'relu')(inputs)
        encoded = Dense(5, activation = 'relu')(encoded)
        encoded = Dense(4, activation = 'relu')(encoded)
        model = Model(inputs, encoded)
        self.encoder = model
        return model
    
    def _decoder(self):
        inputs = Input(shape=(4,))
        decoded = Dense(5, activation = 'relu')(inputs)
        decoded = Dense(6, activation = 'relu')(decoded)
        decoded = Dense(self.w, activation = 'relu')(decoded)
        model = Model(inputs, decoded)
        self.decoder = model
        return model
    
    def encode_decoder(self):
        ec = self._encoder()
        dc = self._decoder()
        
        inputs = Input(shape=(self.w, ))
        ec_out = ec(inputs)
        dc_out = dc(ec_out)
        model = Model(inputs, dc_out)
        
        self.model = model
        return model

    def fit(self, train_x, train_y, test_x, test_y):
        self.model.compile(optimizer='sgd', loss='mse')
        self.model.fit(train_x,train_y,batch_size=32,shuffle='true',epochs=50,validation_data=(test_x,test_y),verbose=1)
    
    def save(self):
        if not os.path.exists(r'./weights'):
            os.mkdir(r'./weights')
        else:
            self.encoder.save(r'./weights/encoder_weights.h5')
            self.decoder.save(r'./weights/decoder_weights.h5')
            self.model.save(r'./weights/ae_weights.h5')