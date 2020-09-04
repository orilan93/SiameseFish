"""
Helpful utility functions.
"""

from tensorflow.python.keras.layers.convolutional import Conv2D
import numpy as np


def get_conv_layers(model):
    """Gets the convolutional layers of a model."""
    current_layer = model
    conv_layers = []
    if hasattr(current_layer, 'layers'):
        current_layer = current_layer.layers
        for layer in current_layer:
            if isinstance(layer, Conv2D):
                conv_layers.append(layer)
            returned_list = get_conv_layers(layer)
            conv_layers.extend(returned_list)
    return conv_layers


def array_distance(arr1, arr2):
    """Gets the distance between elements in two arrays."""
    size = len(arr1)
    arr3 = np.zeros(size)
    for i in range(size):
        arr3[i] = np.abs(arr1[i] - arr2[i])
    return arr3


def groupby_y(x, y):
    """Groups x by y. Assumes that y is sorted."""
    grouped = []
    for i, v in enumerate(y):
        if grouped and y[i - 1] == y[i]:
            grouped[-1].append(x[i])
        else:
            grouped.append([x[i]])
    return grouped


def lr_scheduler(epoch, lr):
    if epoch < 10:
        return 1e-3
    elif epoch < 20:
        return 1e-4
    else:
        return 1e-5
