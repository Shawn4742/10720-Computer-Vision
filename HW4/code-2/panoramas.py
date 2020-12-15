import cv2
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from planarH import ransacH
from BRIEF import briefLite, briefMatch, plotMatches


def imageStitching(im1, im2, H2to1):
    """
    Returns a panorama of im1 and im2 using the given 
    homography matrix

    INPUT
        Warps img2 into img1 reference frame using the provided warpH() function
        H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear
                 equation
    OUTPUT
        Blends img1 and warped img2 and outputs the panorama image
    """
    #######################################
    im1_pano = np.zeros((im1.shape[0] + 80, im1.shape[1] + 750, 3), dtype=np.uint8)
    im1_pano[: im1.shape[0], : im1.shape[1], : im1.shape[2]] = im1
    im1_pano_mask = im1_pano > 0

    # TODO ...
    # warp im2 onto pano
    pano_im = cv2.warpPerspective(im2, H2to1, (im1_pano.shape[1], im1_pano.shape[0]))
    pano_im_mask = pano_im > 0

    # TODO
    # dealing with the center where images meet.
    im_center_mask = np.logical_and(im1_pano_mask, pano_im_mask)

    im_full = pano_im + im1_pano

    im_R = im_full * np.logical_not(im1_pano_mask)
    im_L = im_full * np.logical_not(pano_im_mask)
    # TODO produce im center, mix of pano_im and im1_pano
    im_center = im1_pano * im_center_mask 

    # return im1_pano
    return im_R + im_L + im_center


def imageStitching_noClip(im1, im2, H2to1):
    """
    Returns a panorama of im1 and im2 using the given 
    homography matrix without cliping.
    """
    ######################################
    # TO DO ...
    s = 1
    tx = 0
    # clip
    # establish corners
    im2_H, im2_W, _ = im2.shape

    # The sequence for H is (x, y, 1)
    im2_corners = np.array([ [0,0,1], [im2_W,0,1], [0,im2_H,1], [im2_W,im2_H,1] ]).T
    
    # create new corners
    transfer_corners = H2to1.dot(im2_corners)
    transfer_corners = transfer_corners / transfer_corners[-1,:]
    transfer_corners = np.round(transfer_corners).astype(int)

    im1_H, im1_W, _ = im1.shape

    # transfer_corners: (W, H, 1)
    H_min = np.minimum( 0, np.minimum(transfer_corners[1][0], transfer_corners[1][1]) )
    H_max = np.maximum(im1_H, np.maximum( transfer_corners[1][2], transfer_corners[1][3] ))

    W_min = np.minimum( 0, np.minimum(transfer_corners[0][0], transfer_corners[0][2]) )
    W_max = np.maximum(im1_H, np.maximum( transfer_corners[0][1], transfer_corners[0][3] ))

    H = H_max - H_min
    W = W_max - W_min
    out_size = (W, H)

    print('size of pano:', (H_min, H_max, W_min, W_max))
    # ty = ... used for M_translate matrix
    ty = -H_min 
    # tx = -W_min

    # you actually dont need to use M_scale for the pittsburgh city stitching.
    M_scale = np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]], dtype=np.float64)
    M_translate = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)

    # TODO fill in the arguments
    pano_im2 = cv2.warpPerspective(im2, np.matmul(M_translate, H2to1), out_size)
    pano_im1 = cv2.warpPerspective(im1, M_translate, out_size)

    im1_pano_mask = pano_im1 > 0
    im2_pano_mask = pano_im2 > 0

    # cv2.imshow("panoramas", pano_im2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # TODO
    # should be same line as what you implemented in line 32, in imagestitching
    im_center_mask = np.logical_and(im1_pano_mask, im2_pano_mask)
    pano_im_full = pano_im1 + pano_im2

    im_R = pano_im_full * np.logical_not(im1_pano_mask)
    im_L = pano_im_full * np.logical_not(im2_pano_mask)
    # should be same line as what you implemented in line 39, in imagestitching
    im_center = pano_im1 * im_center_mask
    return im_center + im_R + im_L


def generatePanorama(im1, im2):
    H2to1 = np.load("bestH.npy")
    pano_im = imageStitching_noClip(im1, im2, H2to1)
    return pano_im


if __name__ == "__main__":
    im1 = cv2.imread("../data/incline_L.png")
    im2 = cv2.imread("../data/incline_R.png")

    im1 = cv2.imread("../data/hi_L.jpg")
    im2 = cv2.imread("../data/hi_R.jpg")

    s = 20
    im1 = cv2.resize(im1, (im1.shape[1]//s, im1.shape[0]//s), interpolation = cv2.INTER_AREA)
    im2 = cv2.resize(im2, (im2.shape[1]//s, im2.shape[0]//s), interpolation = cv2.INTER_AREA)
    print(im1.shape)

    # cv2.imshow("panoramas", im2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # swap im1 and im2
    # homography from incline R onto incline L
    # from locs1 to locs2
    locs1, desc1 = briefLite(im2)
    locs2, desc2 = briefLite(im1)
    matches = briefMatch(desc1, desc2)
    # plotMatches(im1,im2,matches,locs1,locs2)


    H2to1 = ransacH(matches, locs1, locs2, num_iter=20000, tol=5)
    # pano_im = imageStitching(im1, im2, H2to1)

    # TODO
    # save bestH.npy
    # np.save("bestH.npy", H2to1)
    # pano_im = generatePanorama(im1, im2)

    pano_im = imageStitching_noClip(im1, im2, H2to1)
    print(H2to1)
    cv2.imwrite("../results/7_3_2.png", pano_im)
    cv2.imshow("panoramas", pano_im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
