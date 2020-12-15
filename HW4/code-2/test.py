import cv2
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from planarH import ransacH
from BRIEF import briefLite, briefMatch, plotMatches

s = 1
tx = 0
ty = -500

im1 = cv2.imread("../data/incline_L.png")
im1_pano = np.zeros((im1.shape[0] + 80, im1.shape[1] + 750, 3), dtype=np.uint8)
im1_pano[: im1.shape[0], : im1.shape[1], : im1.shape[2]] = im1
im1_pano_mask = im1_pano > 0
M_translate = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)

pano_im = cv2.warpPerspective(im1, M_translate, (im1_pano.shape[1], im1_pano.shape[0]))
cv2.imshow("panoramas", pano_im)
cv2.waitKey(0)
cv2.destroyAllWindows()

