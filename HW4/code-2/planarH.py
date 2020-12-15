import numpy as np
import cv2
from BRIEF import briefLite, briefMatch


def computeH(p1, p2):
    """
    INPUTS:
        p1 and p2 - Each are size (2 x N) matrices of corresponding (x, y)'  
                 coordinates between two images
    OUTPUTS:
     H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear 
            equation
    """
    assert p1.shape[1] == p2.shape[1]
    assert p1.shape[0] == 2
    #############################
    # H * p1 -> p2

    # TO DO ...
    N = p1.shape[1]

    u1 = p1[0,:]
    u2 = p1[1,:]

    x1 = p2[0,:]
    x2 = p2[1,:]

    u1 = np.expand_dims(u1, 1)
    u2 = np.expand_dims(u2, 1)
    x1 = np.expand_dims(x1, 1)
    x2 = np.expand_dims(x2, 1)


    first_lines  = np.hstack( (u1, u2, np.ones((N,1)), np.zeros((N,3)), -np.multiply(u1,x1), -np.multiply(u2,x1), -x1 ) )
    second_lines = np.hstack( (np.zeros((N,3)), u1, u2, np.ones((N,1)), -np.multiply(u1,x2), -np.multiply(u2,x2), -x2 ) )
    A = np.vstack((first_lines, second_lines))
    
    u, s, vh = np.linalg.svd(A)
    H2to1 = vh[-1,:].reshape(3,3)

    return H2to1


def ransacH(matches, locs1, locs2, num_iter=5000, tol=2):
    """
    Returns the best homography by computing the best set of matches using
    RANSAC
    INPUTS
        locs1 and locs2 - matrices specifying point locations in each of the images
        matches - matrix specifying matches between these two sets of point locations
        nIter - number of iterations to run RANSAC
        tol - tolerance value for considering a point to be an inlier

    OUTPUTS
        bestH - homography matrix with the most inliers found during RANSAC
    """
    ###########################
    # TO DO ...
    p1 = locs1[matches[:,0]]
    p2 = locs2[matches[:,1]]

    p1 = np.transpose(p1[:,:2])
    p2 = np.transpose(p2[:,:2])

    N = p1.shape[1]
    inlier_max = 0
    p1_homo = np.concatenate(( p1, np.ones((1,N)) ))
    p2_homo = np.concatenate(( p2, np.ones((1,N)) ))
    
    for i in range(int(num_iter)):
        idx = np.random.choice(N, 4, replace=False)
        
        # idx = np.array([0,1,2,3])
        # print(p1[:,idx])
        H = computeH(p1[:,idx], p2[:,idx])

        # check H
        p2_match = np.dot(H, p1_homo)
        p2_match = p2_match / p2_match[-1, :]

        diff = np.linalg.norm(p2_match-p2_homo, axis=0)
        inlier = np.where(diff < tol)[0].shape[0]

        if inlier > inlier_max:
            bestH = H
            inlier_max = inlier

    print('max num of inliers:', inlier_max)
    return bestH


def compositeH(H, template, img):
    """
    Returns final warped harry potter image. 
    INPUTS
        H - homography 
        template - desk image
        img - harry potter image
    OUTPUTS
        final_img - harry potter on book cover image  
    """
    # TODO
    warp_img = cv2.warpPerspective(img, H, (template.shape[1],template.shape[0]))
    idx = np.where(warp_img > 0)

    template[idx] = 0
    final_img = warp_img + template
    return final_img


if __name__ == "__main__":
    im1 = cv2.imread("../data/model_chickenbroth.jpg")
    im2 = cv2.imread("../data/chickenbroth_01.jpg")
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    ransacH(matches, locs1, locs2, num_iter=5000, tol=2)

