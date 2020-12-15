import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import BRIEF

# TODO
compareX, compareY = BRIEF.makeTestPattern()
im1 = cv2.imread('../data/model_chickenbroth.jpg')
locs1, desc1 = BRIEF.briefLite(im1)

H, W = im1.shape[:2]

angle_sets = np.arange(0, 360, 10)
num = []
for angle in angle_sets:
    rot_mat =  cv2.getRotationMatrix2D( (H//2, W//2), angle, 1 )
    im2 = cv2.warpAffine(im1, rot_mat, (H, W))

    locs2, desc2 = BRIEF.briefLite(im2)
    matches = BRIEF.briefMatch(desc1, desc2)
    # BRIEF.plotMatches(im1,im2,matches,locs1,locs2)
    num.append(len(matches))

    print('angle:', angle)

fig = plt.figure()
plt.bar(angle_sets, num, width=5)
plt.xlabel('angle')
plt.ylabel('counts')
plt.show()
