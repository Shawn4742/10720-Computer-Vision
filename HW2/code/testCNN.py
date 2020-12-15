import numpy as np
import multiprocessing
import threading
import queue
import os,time
import torch
import skimage.transform
import torchvision.transforms
import util
import network_layers
import deep_recog
import skimage.transform
from skimage import data
from skimage.transform import resize
from skimage import io

# image = data.camera()
# img = resize(image, (20, 20))
# print(img)

vgg16_weights = util.get_VGG16_weights()
# print(len(vgg16_weights))
# # print(vgg16_weights[0])

# for i in range(len(vgg16_weights)):
#     name = vgg16_weights[i][0]
#     if name == 'maxpool2d':
#         print(vgg16_weights[i][1].shape)
#         print(vgg16_weights[i][2].shape)



path_img = "../data/aquarium/sun_aztvjgubyrgvirup.jpg"
image = io.imread(path_img)
image = image.astype('float')/255

img = deep_recog.preprocess_image(image)
diff = deep_recog.evaluate_deep_extractor(img, vgg16_weights )
print(diff)