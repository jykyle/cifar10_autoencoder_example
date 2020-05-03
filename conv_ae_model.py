'''
Autoencoder (with CNN, batch normalization, without fully connected layers)
Used in train.py
'''

from keras.layers import Input, Conv2D, BatchNormalization, UpSampling2D
from keras.models import Model


def build_model(inputshape=(32,32,3)):
    
    x_input = Input(shape=inputshape)
    
    x = Conv2D(32, kernel_size=(3,3), strides=1, padding='same', activation='relu')(x_input)
    x = BatchNormalization()(x)
    x = Conv2D(32, kernel_size=(3,3), strides=2, padding='same', activation='relu')(x)
    x = Conv2D(32, kernel_size=(3,3), strides=1, padding='same', activation='relu')(x)
    
    x = UpSampling2D()(x)
    x = Conv2D(32, kernel_size=(3,3), strides=1, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x_output = Conv2D(3, kernel_size=(1,1), strides=1, padding='same', activation='sigmoid')(x)
    
    model = Model(inputs=x_input, outputs=x_output)
    
    return model