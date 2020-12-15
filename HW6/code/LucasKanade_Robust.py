import numpy as np
from scipy.interpolate import RectBivariateSpline

# Objective function for only translation: 
# Eq4: \argmin_c { \sum_x T(x) - I(x+c) }
# Eq10: c* = \argmin_c { \delta_I c - [T(x) - I(x+c)]}

# => c = A^-1 b, where A = \delta_I, b = T(x) - I(x+c)

def scale_brightness(data):
    # scale the brightness
    data = data.astype(np.float)
    H, W, numFrames = data.shape
    data = data.reshape((H*W, numFrames))

    avg_data = np.mean(data, axis=1)
    avg_brightness = np.mean(avg_data)

    avg_data_ratio =  avg_brightness/ avg_data 
    data *= avg_data_ratio[:, None] 
    return data.reshape((H, W, numFrames)).astype(np.uint8)


def LucasKanade_M1(It, It1, rect):
    # Input: 
    #   It: template image
    #   It1: Current image
    #   rect: Current position of the object
    #   (top left, bot right coordinates: x1, y1, x2, y2)
    # Output:
    #   p: movement vector dx, dy
    
    # set up the threshold
    threshold = 0.01875
    maxIters = 100
    p = np.zeros(2)          
    x1, y1, x2, y2 = rect

    # put your implementation here
    It = RectBivariateSpline(np.arange(It.shape[0]), np.arange(It.shape[1]), It)
    It1 = RectBivariateSpline(np.arange(It1.shape[0]), np.arange(It1.shape[1]), It1)
    
    # compute T(x)
    It_w, It_h = np.meshgrid(np.arange(x1, x2), np.arange(y1, y2))
    It_rect = It.ev(It_h, It_w)

    for _ in range(maxIters):

        # compute I(x)
        It1_w, It1_h = np.meshgrid(np.arange(x1, x2)+p[0], np.arange(y1, y2)+p[1])
        It1_rect = It1.ev(It1_h, It1_w)

        # grad_x --> dy direction
        grad_x = It1.ev(It1_h, It1_w, dx=0, dy=1).reshape(-1)
        grad_y = It1.ev(It1_h, It1_w, dx=1, dy=0).reshape(-1)

        A = np.vstack((grad_x, grad_y))
        b = It_rect.reshape(-1) - It1_rect.reshape(-1)

        delta_p = np.linalg.lstsq(A.T, b)[0]
        p += delta_p
        if np.linalg.norm(delta_p) < threshold:
            break
    return p



def LucasKanade_M2(It, It1, rect):
    # Input: 
    #   It: template image
    #   It1: Current image
    #   rect: Current position of the object
    #   (top left, bot right coordinates: x1, y1, x2, y2)
    # Output:
    #   p: movement vector dx, dy
    
    # set up the threshold
    threshold = 0.01875
    maxIters = 100
    p = np.zeros(2)          
    x1, y1, x2, y2 = rect

    # put your implementation here
    It = RectBivariateSpline(np.arange(It.shape[0]), np.arange(It.shape[1]), It)
    It1 = RectBivariateSpline(np.arange(It1.shape[0]), np.arange(It1.shape[1]), It1)
    
    # compute T(x)
    It_w, It_h = np.meshgrid(np.arange(x1, x2), np.arange(y1, y2))
    It_rect = It.ev(It_h, It_w)

    for _ in range(maxIters):

        # compute I(x)
        It1_w, It1_h = np.meshgrid(np.arange(x1, x2)+p[0], np.arange(y1, y2)+p[1])
        It1_rect = It1.ev(It1_h, It1_w)

        # grad_x --> dy direction
        grad_x = It1.ev(It1_h, It1_w, dx=0, dy=1).reshape(-1)
        grad_y = It1.ev(It1_h, It1_w, dx=1, dy=0).reshape(-1)

        A = np.vstack((grad_x, grad_y))
        b = It_rect.reshape(-1) - It1_rect.reshape(-1)

        # To minimize effect of outliers,
        # calculate the mean and vairance of the pixel errors

        mean = np.mean(b, axis=0)
        std = np.std(b, axis=0)

        Aii = np.abs(b - mean) - std

        # give high weights within one std
        # give low weights out of one std
        Aii[Aii >= 0] = 1.5
        Aii[Aii < 0] = 0.1

        Aii = np.diag(Aii)
        A = Aii @ A.T
        b = Aii @ b 

        delta_p = np.linalg.lstsq(A, b)[0]
        p += delta_p
        if np.linalg.norm(delta_p) < threshold:
            break
    return p