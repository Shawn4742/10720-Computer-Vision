import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
from mpl_toolkits.axes_grid1 import ImageGrid

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation

from nn import *
from q4 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


d = {}
d['01_list.jpg'] = ['TODOLIST', '1MAKEATODOLIST', '2CHECKOFFTHEFIRST', 'THINGONTODOLIST','3REALIZEYOUHAVEALREADY', 
            'COMPLETEED2THINGS', '4REWARDYOURSELFWITH', 'ANAP']
d['02_letters.jpg'] = ['ABCDEFG', 'HIJKLMN', 'OPQRSTU', 'VWXYZ', '1234567890']
d['03_haiku.jpg'] = ['HAIKUSAREEASY', 'BUTSOMETIMESTHEYDONTMAKESENSE', 'REFRIGERATOR']
d['04_deep.jpg'] = ['DEEPLEARNING', 'DEEPERLEARNING', 'DEEPESTLEARNING']
acc = 0
total = 0

for img in os.listdir('../images'):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
    bboxes, bw = findLetters(im1)

    # plt.imshow(bw)
    # for bbox in bboxes:
    #     minr, minc, maxr, maxc = bbox
    #     rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
    #                             fill=False, edgecolor='red', linewidth=2)
    #     plt.gca().add_patch(rect)
    # plt.show()
    # plt.savefig("out_" + img + ".jpg")
    
    # find the rows using..RANSAC, counting, clustering, etc.
    ##########################
    ##### your code here #####
    ##########################

    # sort by rows, the cols
    # minr, minc, maxr, maxc = bbox
    bboxes = sorted(bboxes, key=lambda x: x[0])

    bboxes = np.array(bboxes)
    height = np.max(bboxes[:,2] - bboxes[:,0]) * 0.7
    one_line_boxes = []
    sorted_boxes = []
    minr_prev = np.min(bboxes[:,0])
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox

        if minr - minr_prev > height:
            one_line_boxes = sorted(one_line_boxes, key=lambda x: x[1])
            sorted_boxes.append(one_line_boxes)
            one_line_boxes = [bbox]
        else:
            one_line_boxes.append(bbox)

        minr_prev = minr
    
    sorted_boxes.append(sorted(one_line_boxes, key=lambda x: x[1]))

    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
    ##########################
    ##### your code here #####
    ##########################
    pad_min = 25
    im_sets = []
    
    for one_line in sorted_boxes:
        one_line_im = []
        for box in one_line:
            minr, minc, maxr, maxc = box

            if maxr - minr > maxc - minc:
                pad_r = pad_min
                pad_c = (maxr - minr - (maxc - minc)) // 2 + pad_min
            else:
                pad_c = pad_min
                pad_r = -(maxr - minr - (maxc - minc)) // 2 + pad_min            

            im = bw[minr: maxr+1, minc:maxc+1]
            im = np.pad(im, ((pad_r,pad_r), (pad_c,pad_c)), 'constant', constant_values=(1,1))
            # print('im shape:', im.shape)
            im = skimage.transform.resize(im, (32, 32)).T

            # but why?
            im = skimage.morphology.erosion(im)
            one_line_im.append(im.flatten())

        im_sets.append(one_line_im)

    if False:
    # if True:
        # fig = plt.figure(1, (6., 8.))
        fig = plt.figure()
        num = 15
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                        nrows_ncols=(1, num),  # creates 2x2 grid of axes
                        axes_pad=0.1,  # pad between axes in inch.
                        )
        
        images = np.array(im_sets[2])
        images = images[:num].reshape(num, 32, 32)

        displayed = images

        for ax, im in zip(grid, displayed):
            ax.imshow(im.T)
        plt.show()
    
    # load the weights
    # run the crops through your neural network and print them out
    import pickle
    import string
    letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    params = pickle.load(open('q3_weights.pickle','rb'))
    ##########################
    ##### your code here #####
    ##########################

    text = d[img]
    count = 0
    
    for one_line in im_sets:
        one_line = np.asarray(one_line)
        h1 = forward(one_line, params,'layer1')
        probs = forward(h1, params,'output',softmax)
        pred_label = np.argmax(probs, axis=1) 

        for i in range(len(pred_label)):
            if text[count][i] == letters[pred_label[i]]:
                acc += 1
            total += 1
        count += 1

        print(''.join(letters[pred_label]))       
    print('acc:', acc/total * 1.0)