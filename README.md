# cifar10_autoencoder_example

Simple autoencoder example on CIFAR-10 dataset, 
using convolutional layers and batch normalization, no fully connected layers.
Implemented with the dataset in directories instead of importing in order to suit, preprocess and train other datasets.

<p>&nbsp;</p>
<b>Major libraries and versions used to build this project:</b>

Python==3.6.8

Keras==2.2.4

tensorflow-gpu==1.14.0

Pillow==6.1.0

numpy==1.18.1

<p>&nbsp;</p>
<b>Example of a run:</b>

python train.py

python evaluate.py

Other python modules are used by train / evaluate.

<p>&nbsp;</p>
<b>Example of using arguments as inputs</b>

python train.py -d cifar_10_dataset/train -o result -p 1 -nbatch 256 -e 20

python evaluate.py -d cifar_10_dataset/test -m result/model.h5 -o result/result.png -p 1 -nx 20 -ny 20
