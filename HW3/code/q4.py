import numpy as np

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation

# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    bboxes = []
    bw = None
    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes 
    # this can be 10 to 15 lines of code using skimage functions

    ##########################
    ##### your code here #####
    ##########################

    # noise
    image = skimage.filters.gaussian(image)

    # denoise
    image = skimage.restoration.denoise_bilateral(image, multichannel=True)

    # greyscale
    image = skimage.color.rgb2gray(image)

    # threshold
    thresh = skimage.filters.threshold_otsu(image)
    # character in black, background in white
    bw = image < thresh

    # morphology
    bw = skimage.morphology.closing(bw)
    bw = skimage.segmentation.clear_border(bw)

    # label
    im = skimage.measure.label(bw)

    bboxes = []
    mean_area = np.mean([region.area for region in skimage.measure.regionprops(im)])
    for region in skimage.measure.regionprops(im):
        if region.area > mean_area / 2.6:
            bboxes.append(region.bbox)

    bw = 1 - bw
    return bboxes, bw.astype(np.float)