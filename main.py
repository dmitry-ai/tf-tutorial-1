# 2019-05-12 https://www.tensorflow.org/tutorials/keras/basic_classification
#from __future__ import absolute_import, division, print_function
''' 2019-05-12
«matplotlib.pyplot is a state-based interface to matplotlib. It provides a MATLAB-like way of plotting.»:
https://matplotlib.org/api/_as_gen/matplotlib.pyplot.html '''
import matplotlib.pyplot as plt
import tensorflow as tf
''' 2019-05-12
https://keras.io
https://github.com/keras-team/keras
https://en.wikipedia.org/wiki/Keras
https://www.tensorflow.org/versions/r1.13/api_docs/python/tf/keras '''
import tensorflow.keras as keras
#import numpy as np
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
3) The output: «(60000, 28, 28)»
It means that the `train_images` array contains 60000 images of 28x28 pixels each. '''
print(train_images.shape)
''' 2019-05-12
1) «Return the length (the number of items) of an object.
The argument may be a sequence (such as a string, bytes, tuple, list, or range) 
or a collection (such as a dictionary, set, or frozen set).»
https://docs.python.org/3.7/library/functions.html#len
2) The output: «60000» '''
print(len(train_labels))
# 2019-05-12 The output: «[9 0 0 ... 3 0 5]»
print(train_labels)
# 2019-05-12 The output: «(10000, 28, 28)»
print(test_images.shape)
# 2019-05-12 The output: «10000»
print(len(test_labels))
'''
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show() '''
train_images = train_images / 255.0
test_images = test_images / 255.0
''' 2019-05-12
«The `Sequential` model is a linear stack of layers»:
https://keras.io/getting-started/sequential-model-guide
https://keras.io/models/sequential '''
model = keras.Sequential([
	# 2019-05-12
	# 1) «Flattens the input»:
	# https://keras.io/layers/core#flatten
	# https://www.tensorflow.org/versions/r1.13/api_docs/python/tf/keras/layers/Flatten
	#
	# 2) «The first layer in this network, `tf.keras.layers.Flatten`, transforms the format of the images
	# from a 2d-array (of 28 by 28 pixels), to a 1d-array of 28 * 28 = 784 pixels.
	# Think of this layer as unstacking rows of pixels in the image and lining them up.
	# This layer has no parameters to learn; it only reformats the data.»
	# https://www.tensorflow.org/tutorials/keras/basic_classification#setup_the_layers
	keras.layers.Flatten(input_shape=(28, 28)),
	# 2019-05-12
	# 1) «Just your regular densely-connected NN layer.
	# Dense implements the operation:
	# 		output = activation(dot(input, kernel) + bias)
	# where:
	# 	`activation` is the element-wise activation function passed as the `activation` argument,
	# 	`kernel` is a weights matrix created by the layer,
	# 	`bias` is a bias vector created by the layer (only applicable if `use_bias` is `True`).
	# Note: if the input to the layer has a rank greater than 2,
	# then it is flattened prior to the initial dot product with kernel.»
	# https://keras.io/layers/core#dense
	# https://www.tensorflow.org/versions/r1.13/api_docs/python/tf/keras/layers/Dense
	#
	# 2) After the pixels are flattened,
	# the network consists of a sequence of two `tf.keras.layers.Dense` layers.
	# These are densely-connected, or fully-connected, neural layers.
	# *) The first `Dense` layer has 128 nodes (or neurons).
	# *) The second (and last) layer is a 10-node softmax layer —
	# this returns an array of 10 probability scores that sum to 1.
	# Each node contains a score
	# that indicates the probability that the current image belongs to one of the 10 classes.
	# https://www.tensorflow.org/tutorials/keras/basic_classification#setup_the_layers
	#
	# 3) `tf.nn.relu`: «Computes rectified linear: max(features, 0)».
	# https://www.tensorflow.org/versions/r1.13/api_docs/python/tf/nn/relu
	#
	# 3) The `activation` argument specifies the activation function to apply to the output of the convolution.
	# Here, we specify ReLU activation with tf.nn.relu.
	# https://www.tensorflow.org/tutorials/estimators/cnn#convolutional_layer_1
	#
	# 4) «In artificial neural networks, the activation function of a node
	# defines the output of that node, or "neuron," given an input or set of inputs.
	# This output is then used as input for the next node and so on
	# until a desired solution to the original problem is found.»:
	# https://en.wikipedia.org/wiki/Activation_function
	#
	# 5) ReLU means «Rectified Linear Unit»:
	# https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks
	#
	# 6) «In the context of artificial neural networks,
	# the rectifier is an activation function defined as the positive part of its argument:
	# 		f(x) = max(0,x)
	# A unit employing the rectifier is also called a rectified linear unit (ReLU).»
	# https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
	keras.layers.Dense(128, activation=tf.nn.relu),
	# 2019-05-12
	# `tf.nn.softmax`:  «Computes softmax activations. This function performs the equivalent of
	# 		softmax = tf.exp(logits) / tf.reduce_sum(tf.exp(logits), axis)
	# »
	# https://www.tensorflow.org/versions/r1.13/api_docs/python/tf/nn/softmax
	keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5)
