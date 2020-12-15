import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanadeAffine(It, It1, rect):
    # Input: 
    #   It: template image, T
    #   It1: Current image, I
    #   rect: Current position of the object
    #   (top left, bot right coordinates: x1, y1, x2, y2)
    # Output:
    #   M: the Affine warp matrix [2x3 numpy array]

    # set up the threshold
    threshold = 0.01875
    maxIters = 100
    p = np.zeros(6)
    x1, y1, x2, y2 = rect

    # put your implementation here
    # h, w = It.shape
    It = RectBivariateSpline(np.arange(It.shape[0]), np.arange(It.shape[1]), It)
    It1 = RectBivariateSpline(np.arange(It1.shape[0]), np.arange(It1.shape[1]), It1)
    
    # compute T(x)
    It_w, It_h = np.meshgrid(np.arange(x1, x2), np.arange(y1, y2))
    It_rect = It.ev(It_h, It_w)

    for _ in range(maxIters):
        # print(i)

        # compute I(x)
        It1_w = (p[0]+1) * It_w + p[2] * It_h + p[4]
        It1_h = p[1] * It_w + (p[3]+1) * It_h + p[5]
        It1_rect = It1.ev(It1_h, It1_w)

        # ### -------------------------------- ###
        # # Bound the box inside the It1?
        # It1_w = It1_w.flatten()
        # It1_h = It1_h.flatten()
        # It_rect = It_rect.flatten()
        # It1_rect = It1_rect.flatten()

        # idx = (It1_w > 0) & (It1_w < w) & (It1_h > 0) & (It1_h < h) 
        # It1_w = It1_w[idx]
        # It1_h = It1_h[idx]

        # It_rect = It_rect[idx]
        # It1_rect = It1_rect[idx]
        # ### -------------------------------- ###

        # grad_x --> dy direction
        grad_x = It1.ev(It1_h, It1_w, dx=0, dy=1).reshape(-1)
        grad_y = It1.ev(It1_h, It1_w, dx=1, dy=0).reshape(-1)

        A = np.zeros((grad_x.shape[0], 6))
        A[:,0] = grad_x * It1_w.flatten()
        A[:,1] = grad_y * It1_w.flatten()

        A[:,2] = grad_x * It1_h.flatten()
        A[:,3] = grad_y * It1_h.flatten()

        A[:,4] = grad_x
        A[:,5] = grad_y

        b = It_rect.reshape(-1) - It1_rect.reshape(-1)

        delta_p = np.linalg.lstsq(A, b)[0]
        p += delta_p
        if np.linalg.norm(delta_p) < threshold:
            break
    
    # reshape the output affine matrix
    # M = np.array([[1.0+p[0], p[1],    p[2]],
    #              [p[3],     1.0+p[4], p[5]]]).reshape(2, 3)

    # make the sequence of p consistent.
    M = np.array([[1.0+p[0], p[2],    p[4]],
                 [p[1],     1.0+p[3], p[5]]]).reshape(2, 3)
    return M
