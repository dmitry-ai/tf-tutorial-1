# 2019-05-12 https://www.tensorflow.org/tutorials/keras/basic_classification
#from __future__ import absolute_import, division, print_function
import tensorflow as tf
import tensorflow.keras as keras
#from tensorflow import keras
#import numpy as np
#import matplotlib.pyplot as plt
#print(tf.__version__)
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = [
	'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]
''' 2019-05-12
1) `train_images` is an N-dimensional array: https://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html
2) The `shape` property returns the current shape of an array:
https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.shape.html
3) The output: (60000, 28, 28)
It means that the `train_images` array contains 60000 images of 28x28 pixels each. 
'''
print(train_images.shape)
