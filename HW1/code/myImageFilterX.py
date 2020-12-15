import cv2
import numpy as np


def myImageFilterX(img0, hfilt):
    # Your implemention

    # pad the image
    hfilt_h = hfilt.shape[0]
    hfilt_w = hfilt.shape[1]
    pad_w = int(np.floor(hfilt_w/2.0))
    pad_h = int(np.floor(hfilt_h/2.0))
    img1 = np.pad(img0, ((pad_w, pad_w), (pad_h, pad_h)), mode='edge')

    # slice the matrix
    # see Ref: 
    # https://zhuanlan.zhihu.com/p/64933417
    sub_shape = (hfilt_h, hfilt_w)
    view_shape = tuple(np.subtract(img1.shape, sub_shape) + 1) + sub_shape
    strides = img1.strides + img1.strides    
    img2 = np.lib.stride_tricks.as_strided(img1, view_shape, strides)

    # vectorize the img2 for last 2 dim
    img2 = img2.reshape(img2.shape[0], img2.shape[1], img2.shape[2]*img2.shape[3])

    # vecrorize hfilt
    # computation tested in test.py
    return np.dot(img2, hfilt.reshape(-1))
