import numpy as np
import cv2
import skimage.io
from BRIEF import briefLite, briefMatch, plotMatches
from planarH import ransacH, compositeH

# warp harry potter onto cv desk image
# save final image as final_image
# TODO

im1 = cv2.imread("../data/pf_scan_scaled.jpg")
im2 = cv2.imread("../data/pf_desk.jpg")
im3 = cv2.imread("../data/hp_cover.jpg")

# im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
# im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
# im3 = cv2.cvtColor(im3, cv2.COLOR_BGR2GRAY)

# to resize im3 is better.
im3 = cv2.resize(im3, (im1.shape[1], im1.shape[0]))
# im1 = cv2.resize(im1, (im3.shape[1], im3.shape[0]))

locs1, desc1 = briefLite(im1)
locs2, desc2 = briefLite(im2)

matches = briefMatch(desc1, desc2)

print(im1.shape)
print(im2.shape)
print(im3.shape)

# plotMatches(im1,im2,matches,locs1,locs2)
num_iter = 5e3
tol = 3
bestH = ransacH(matches, locs1, locs2, num_iter=num_iter, tol=tol)
print('H:', bestH)

final_img = compositeH(bestH, im2, im3)
# final_img = cv2.warpPerspective(im3, bestH, (im2.shape[1],im2.shape[0]))

res = final_img
cv2.imshow("iter=%d_tol=%d" % (num_iter, tol), res)
cv2.waitKey(0)
cv2.destroyAllWindows()