'''
This module is called if model needs to be compiled for fit/evaluate (mainly for train.py)
This script is to be expanded if other optimizer functions are going to be used.
'''

import keras
    
def Adam(model, learn_rate=0.001):
    
    opt = keras.optimizers.Adam(lr=learn_rate)
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])
    
    return model