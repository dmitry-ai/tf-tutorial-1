# 2019-05-12 https://www.tensorflow.org/tutorials/keras/basic_classification
#from __future__ import absolute_import, division, print_function
import tensorflow as tf
import tensorflow.keras as keras
''' 2019-05-12
«matplotlib.pyplot is a state-based interface to matplotlib. It provides a MATLAB-like way of plotting.»:
https://matplotlib.org/api/_as_gen/matplotlib.pyplot.html '''
import matplotlib.pyplot as plt
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
«Create a new figure.
	figsize : (float, float), optional, default: None
		width, height in inches. 
		If not provided, defaults to rcParams["figure.figsize"] = [6.4, 4.8].»
https://matplotlib.org/api/_as_gen/matplotlib.pyplot.figure.html#matplotlib.pyplot.figure '''
plt.figure(figsize=(10,10))
''' 2019-05-12
1) «Rather than being a function, `range` is actually an immutable sequence type»:
https://docs.python.org/3.7/library/functions.html#func-range
2) «The range type represents an immutable sequence of numbers
and is commonly used for looping a specific number of times in for loops».
>>> list(range(10))
	[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
https://docs.python.org/3.7/library/stdtypes.html#typesseq-range '''
for i in range(25):
	plt.subplot(5,5,i+1)
	plt.xticks([])
	plt.yticks([])
	plt.grid(False)
	plt.imshow(train_images[i], cmap=plt.cm.binary)
	plt.xlabel(class_names[train_labels[i]])
plt.show()
