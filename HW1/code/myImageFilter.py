import cv2
import numpy as np



def myImageFilter(img0, hfilt):
    # Your implemention

    # pad the image
    hfilt_h = hfilt.shape[0]
    hfilt_w = hfilt.shape[1]
    pad_w = int(np.floor(hfilt_w/2.0))
    pad_h = int(np.floor(hfilt_h/2.0))
    img1 = np.pad(img0, ((pad_w, pad_w), (pad_h, pad_h)), mode='edge')
    # img1 = np.pad(img0, ((pad_w, pad_w), (pad_h, pad_h)), mode='constant', constant_values = 0)

    hfilt = np.flip(hfilt)
    img2 = np.zeros(img0.shape)
    for i in range(img0.shape[0]):
    	for j in range(img0.shape[1]):
    		img_temp = img1[ i: i+hfilt_w, j: j+hfilt_h ] 
    		img2[i][j] = np.multiply(img_temp, hfilt).sum()

    return img2
