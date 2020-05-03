'''
Main training module for autoencoder.
Preprocess the raw data -> build model -> train model -> drop weights and learning curve
This training module will also work for dataset with format: parent_directory/label(1..n)/data_in_image_format
'''

import preprocess_data
import conv_ae_model
import compile_utils
import image_plot_utils

import keras
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

# optional : recommended for using only part of GPU while training
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
KTF.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))


def train_main(train_directory, num_pool, output_directory, nbatch=128, epoch=10):
    
    # preprocess data and build model
    train_data, _ = preprocess_data.preprocess(train_directory, num_pool)
    ae_model = conv_ae_model.build_model(train_data.shape[1:])
    ae_model = compile_utils.Adam(ae_model)
    
    # train model
    hist = ae_model.fit(train_data, train_data, batch_size=nbatch, epochs=epoch)
    
    # output weight and learning curve
    os.makedirs('./{}'.format(output_directory), exist_ok=True)
    ae_model.save('{}/model.h5'.format(output_directory))
    image_plot_utils.learning_curve_plot(hist, output_directory, 'loss')
    image_plot_utils.learning_curve_plot(hist, output_directory, 'acc')
    print('End of train. Results saved in: {}'.format(output_directory))
    

if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description = 'Arguments required for training model (optimized for cifar-10 data)')    
    parser.add_argument('-d', '--datadir', type=str, help='Define input data directory', default='cifar_10_dataset/train')
    parser.add_argument('-o', '--outputdir', type=str, help='Define location of where output files should go', default='result')
    parser.add_argument('-p', '--pool', type=int, help='Define the number of processes (in CPU) to run preprocess on', default=10)
    parser.add_argument('-nb', '--nbatch', type=int, help='Define batch size when training', default=128)
    parser.add_argument('-e', '--epoch', type=int, help='Define number of epochs when training', default=10)
    
    args = parser.parse_args()
    
    train_main(train_directory=args.datadir, num_pool=args.pool, output_directory=args.outputdir, nbatch=args.nbatch, epoch=args.epoch)