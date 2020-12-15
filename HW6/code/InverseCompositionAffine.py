import numpy as np
from scipy.interpolate import RectBivariateSpline

def InverseCompositionAffine(It, It1, rect):
    # Input: 
    #   It: template image
    #   It1: Current image
    #   rect: Current position of the object
    #   (top left, bot right coordinates: x1, y1, x2, y2)
    # Output:
    #   M: the Affine warp matrix [2x3 numpy array]

    # set up the threshold
    threshold = 0.01875
    maxIters = 100
    p = np.zeros((6,1))
    x1,y1,x2,y2 = rect

    # put your implementation here

    ### pre-compute
    # h, w = It.shape
    It = RectBivariateSpline(np.arange(It.shape[0]), np.arange(It.shape[1]), It)
    It1 = RectBivariateSpline(np.arange(It1.shape[0]), np.arange(It1.shape[1]), It1)

    # compute T(x)
    It_w, It_h = np.meshgrid(np.arange(x1, x2+0.5), np.arange(y1, y2+0.5))
    It_rect = It.ev(It_h, It_w)

    # grad_x --> dy direction
    grad_x = It.ev(It_h, It_w, dy=1).reshape(-1)
    grad_y = It.ev(It_h, It_w, dx=1).reshape(-1) 

    J = np.zeros((grad_x.shape[0], 6))
    J[:,0] = grad_x * It_w.flatten()
    J[:,1] = grad_y * It_w.flatten()

    J[:,2] = grad_x * It_h.flatten()
    J[:,3] = grad_y * It_h.flatten()

    J[:,4] = grad_x
    J[:,5] = grad_y

    H = J.T @ J
    inv_H = np.linalg.inv(H + np.eye(H.shape[0])* 1e-16) @ J.T


    M = np.array([[1.0+p[0], p[2],    p[4]],
                [p[1],     1.0+p[3], p[5]]]).reshape(2, 3)
    M = np.vstack((M, np.array([0,0,1])))

    ### iteration
    for _ in range(maxIters):
        It1_w = (p[0]+1) * It_w + p[2] * It_h + p[4]
        It1_h = p[1] * It_w + (p[3]+1) * It_h + p[5]
        It1_rect = It1.ev(It1_h, It1_w)

        E = It1_rect.reshape(-1) - It_rect.reshape(-1)
        delta_p = inv_H @ E
        p_norm = np.linalg.norm(delta_p)

        # # reshape the output affine matrix
        # M = np.array([[1.0+p[0], p[1],    p[2]],
        #              [p[3],     1.0+p[4], p[5]]]).reshape(2, 3)
        
        # make the sequence of p consistent.


        P2 = np.array([[1.0+delta_p[0], delta_p[2],    delta_p[4]],
                    [delta_p[1],     1.0+delta_p[3], delta_p[5]]]).reshape(2, 3)
        # P2 = np.array([[delta_p[0], delta_p[2],    delta_p[4]],
        #             [delta_p[1],     delta_p[3], delta_p[5]]]).reshape(2, 3)
        P2 = np.vstack((P2, np.array([0,0,1])))
        M = M @ np.linalg.inv(P2)

        p = M[:2].reshape((6,1), order='F')
        p[0] -= 1
        p[3] -= 1
        if p_norm < threshold:
            break

    return M[:2]
