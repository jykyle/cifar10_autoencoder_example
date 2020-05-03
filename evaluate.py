
import preprocess_data
import image_plot_utils

import keras
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import argparse

# optional : recommended for using only part of GPU while training
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
KTF.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))


def evaluate_main(test_directory, model_file, result_file, num_pool, numx, numy):
    
    # load model and preprocess data for predict, then predict
    model = load_model(model_file)
    test_data, _ = preprocess_data.preprocess(test_directory, num_pool)
    pred_data = model.predict(x=test_data)
    
    # evaluate MSE
    mean_squared_error = image_plot_utils.calc_mse(pred_data, test_data)
    print('Mean squared error of predicted data compared to real data :', mean_squared_error)
    
    # this is to add variety to the resulting plot file to observe the performance of autoencoder on various labels
    randomize = np.arange(test_data.shape[0])
    np.random.shuffle(randomize)
    test_data = test_data[randomize]
    pred_data = pred_data[randomize]
    
    # then output the specifed number of images cropped together as png file
    image_plot_utils.print_images(test_data, pred_data, save_file=result_file, num_x=numx, num_y=numy)
    print('Image output displayed at: {}'.format(result_file))


if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description = 'Arguments required for training model (optimized for cifar-10 data)')    
    parser.add_argument('-d', '--datadir', type=str, help='Define data directory to test', default='cifar_10_dataset/test')
    parser.add_argument('-m', '--model', type=str, help='Define model file directory', default='result/model.h5')
    parser.add_argument('-o', '--output', type=str, help='Define location of where output image should go', default='result/result.png')
    parser.add_argument('-p', '--pool', type=int, help='Define the number of processes (in CPU) to run preprocess on', default=10)
    parser.add_argument('-nx', '--numx', type=int, help='Define number of images to put on the output image: horizontal axis', default=10)
    parser.add_argument('-ny', '--numy', type=int, help='Define number of images to put on the output image: vertical axis', default=20)
    args = parser.parse_args()
    
    evaluate_main(test_directory=args.datadir, model_file=args.model, result_file=args.output, num_pool=args.pool, numx=args.numx, numy=args.numy)