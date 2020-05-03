'''
Utility methods for train / evaluate.py
'''

import numpy as np
import matplotlib.pyplot as plt

# Plots and saves image file of given array of pixel values in RGB format
def imageplot(numpy_arr, save_file):
    x_size = np.max([6, numpy_arr.shape[1] / 100])
    y_size = np.max([4, numpy_arr.shape[0] / 100])
    plt.figure(figsize=(x_size,y_size)) # define size of the plot flexible to the size of the input image array
    plt.imshow(numpy_arr)
    plt.axis('off')
    plt.tight_layout() # reduce the amount of whitespace on the sides
    plt.savefig(save_file)
    plt.clf()

# Given true and predicted data of autoencoder, saves image file to custom directory,
# define number of data to display in the output image file
def print_images(true, pred, save_file, num_x=10, num_y=10):
    y_x_images = None
    for i in range(num_y):
        x_images = None
        for j in range(num_x):
            # add images horizontally, true on the left, predicted on the right (column-wise stacking)
            if x_images is None:
                x_images = np.hstack((true[i*num_x+j], pred[i*num_x+j]))
            else:
                x_images = np.hstack((x_images, true[i*num_x+j], pred[i*num_x+j]))
        if y_x_images is None:
            y_x_images = x_images
        else:
            # insert horizontally joined images on the next row (row-wise stacking)
            y_x_images = np.vstack((y_x_images, x_images))
    
    # finished construction of a 'big' array of picture data, need to plot and save this array as image
    imageplot(y_x_images, save_file)


# calculate the MSE of predicted and real data for evaluation
def calc_mse(pred, true):
    return ((pred - true) ** 2).mean()


# plots learning curve after training (loss / acc)
def learning_curve_plot(history, outdir, metric):
    plt.plot(history.history[metric])
    plt.title('Model {}'.format(metric))
    plt.ylabel(metric)
    plt.xlabel('epoch')
    plt.savefig('{}/train_{}.png'.format(outdir, metric))
    plt.clf()