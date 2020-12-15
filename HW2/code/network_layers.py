import numpy as np
import scipy.ndimage
import os
import skimage.measure

def extract_deep_feature(x, vgg16_weights):
    '''
    Extracts deep features from the given VGG-16 weights.

    [input]
    * x: numpy.ndarray of shape (H, W, 3)
    * vgg16_weights: list of shape (L, 3)

    [output]
    * feat: numpy.ndarray of shape (K)
    '''
    # x_shape = x.shape
    # print('x shape:', x_shape)
    x = x.astype(np.float64)
    count = 0
    for layer in vgg16_weights:
        name = layer[0]
        if name == 'conv2d':
            w, b = layer[1], layer[2]
            x = multichannel_conv2d(x, w, b)
        elif name == 'relu':
            x = relu(x)
        elif name == 'maxpool2d':
            size = layer[1]
            x = max_pool2d(x, size)
        elif name == 'linear':
            w, b = layer[1], layer[2]
            x = linear(x, w, b)

            count += 1
            if count == 2:
                return x
        else:
            print('layers not found!')

    # return x

def multichannel_conv2d(x, weight, bias):
    '''
    Performs multi-channel 2D convolution.

    [input]
    * x: numpy.ndarray of shape (H, W, input_dim)
    * weight: numpy.ndarray of shape (output_dim, input_dim, kernel_size, kernel_size)
    * bias: numpy.ndarray of shape (output_dim)

    [output]
    * feat: numpy.ndarray of shape (H, W, output_dim)
    '''
    # input_dim --> image channel
    # output_dim --> num of filters
    output_dim, input_dim = weight.shape[0], weight.shape[1]

    weight = weight.astype(np.float64)
    bias = bias.astype(np.float64)

    feat = []
    for num in range(output_dim):
        img_channel = []
        for c in range(input_dim):
            w = np.flip(weight[num, c, :, :])
            # w = weight[num, c, :, :]
            img = scipy.ndimage.convolve(x[:,:,c], w, mode='nearest', cval=0.0)
            img_channel.append(img)
        img_filter = sum(img_channel) + bias[num]
        feat.append(img_filter)
    
    feat = np.dstack(feat)
    return feat

def relu(x):
    '''
    Rectified linear unit.

    [input]
    * x: numpy.ndarray

    [output]
    * y: numpy.ndarray
    '''

    return np.maximum(x, np.zeros_like(x))

def max_pool2d(x, size):
    '''
    2D max pooling operation.

    [input]
    * x: numpy.ndarray of shape (H, W, input_dim)
    * size: pooling receptive field

    [output]
    * y: numpy.ndarray of shape (H/size, W/size, input_dim)
    '''

    # _, _, channels = x.shape

    # y = []
    # for c in range(channels):
    #     img = skimage.measure.block_reduce(x[:,:,c], (size,size), np.max)
    #     y.append(img)
    # y = np.dstack(y)
    # return y
    return skimage.measure.block_reduce(x, (size, size, 1), np.max)
        

def linear(x,W,b):
    '''
    Fully-connected layer.

    [input]
    * x: numpy.ndarray of shape (input_dim)
    * weight: numpy.ndarray of shape (output_dim,input_dim)
    * bias: numpy.ndarray of shape (output_dim)

    [output]
    * y: numpy.ndarray of shape (output_dim)
    '''
    W = W.astype(np.float64)
    b = b.astype(np.float64)

    x = x.flatten()
    # return np.dot(W, x) + b
    return x.dot(W.T) + b

