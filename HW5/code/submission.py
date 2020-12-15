# ##################################################################### #
# 16720B: Computer Vision Homework 5
# Carnegie Mellon University
# Oct. 26, 2020
# ##################################################################### #


# Insert your package here
from skimage.color import rgb2xyz
from scipy.sparse import kron as spkron
from scipy.sparse import eye as speye
from scipy.sparse.linalg import lsqr as splsqr
import pdb
from utils import integrateFrankot

import numpy as np
import helper
from scipy.ndimage.filters import gaussian_filter
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
import utils
from scipy.sparse.linalg import lsqr

'''
Q3.2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''
def eightpoint(pts1, pts2, M):
    # Replace pass by your implementation
    pts1 = pts1/ float(M)
    pts2 = pts2/ float(M)

    A = np.vstack( (
        pts1[:,0] * pts2[:,0], pts1[:,0] * pts2[:,1], pts1[:,0], 
        pts1[:,1] * pts2[:,0], pts1[:,1] * pts2[:,1], pts1[:,1], 
        pts2[:,0], pts2[:,1], np.ones(pts1.shape[0])
    ) )   

    # A = A.T
    u, s, v = np.linalg.svd(A.dot(A.T))
    F = v[-1,:].reshape((3,3), order='F')

    # u, s, v = np.linalg.svd(A.T)
    # F = v[-1,:].reshape(3,3)
    
    T = np.diag([1.0/M, 1.0/M, 1.0])

    F = helper.refineF(F, pts1, pts2)
    F = np.dot(T.T, F).dot(T)
    return F


'''
Q3.2.2: Seven Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: Farray, a list of estimated fundamental matrix.
'''
def sevenpoint(pts1, pts2, M):
    # Replace pass by your implementation
    pts1 = pts1/ float(M)
    pts2 = pts2/ float(M)
    print(pts1.shape[0])

    A = np.vstack( (
        pts1[:,0] * pts2[:,0], pts1[:,0] * pts2[:,1], pts1[:,0], 
        pts1[:,1] * pts2[:,0], pts1[:,1] * pts2[:,1], pts1[:,1], 
        pts2[:,0], pts2[:,1], np.ones(pts1.shape[0])
    ) )   

    # A = A.T
    u, s, v = np.linalg.svd(A.dot(A.T))

    v1 = v[-1, :]
    v2 = v[-2, :]
    F1 = v1.reshape((3,3), order='F')
    F2 = v2.reshape((3,3), order='F')

    # compute coefficients
    # a0 + a1 x + a2 x^2 + a3 x^3 = 0
    f = lambda x: np.linalg.det( x*F1 + (1-x)*F2 )
    a0 = f(0)
    a2 = (f(1) - a0 + f(-1) - a0) / 2.0
    a3 =  ((f(2)-f(-2)) - 2*(f(1)-f(-1))) / 12.0
    a1 = f(0) - a0 - a2 - a3

    roots = np.roots([a3, a2, a1, a0])
    T = np.diag([1.0/M, 1.0/M, 1.0])

    Fs = []
    for root in roots:
        if np.isreal(root):
            root = np.real(root)
            F = root * F1 + (1-root) * F2
            # F = helper.refineF(F, pts1, pts2)
            F = np.dot(T.T, F).dot(T)
            Fs.append(F)

    return Fs





'''
Q3.3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):
    # Replace pass by your implementation

    # F = (K')^{-1} E K^{-1}
    return K2.T @ F @ K1


'''
Q3.3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''

def triangulate(C1, pts1, C2, pts2):
    # Replace pass by your implementation
    N = pts1.shape[0]
    P = np.zeros((N, 3+1))

    for i in range(N):
        x1 = pts1[i, 0]
        y1 = pts1[i, 1]
        x2 = pts2[i, 0]
        y2 = pts2[i, 1]

        A = np.vstack( (
            y1 * C1[2,:] - C1[1,:], 
            C1[0,:] - x1 * C1[2, :], 
            y2 * C2[2,:] - C2[1,:],
            C2[0,:] - x2 * C2[2, :]
        ) )       

        u, s, v = np.linalg.svd(A.T @ A)
        P[i, :] = v[-1, :] / v[-1, -1]

    # find errors
    p1_proj = C1 @ P.T
    p2_proj = C2 @ P.T

    p1_proj[:2,:] = p1_proj[:2,:] / p1_proj[-1,:]
    p2_proj[:2,:] = p2_proj[:2,:] / p2_proj[-1,:]

    # err = np.linalg.norm(p1_proj[:2,:].T - pts1, ord=2) + np.linalg.norm(p2_proj[:2,:].T - pts2, ord=2)
    err = np.sum( np.square(p1_proj[:2,:].T - pts1) + np.square(np.linalg.norm(p2_proj[:2,:].T - pts2)) )
    # P: Nx4 --> Nx3
    return P[:, :3], err


'''
Q3.4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''
def g_kernel(size, sigma):
    '''
    reference: https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy
    '''
    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
    return kernel / np.sum(kernel)

def epipolarCorrespondence(im1, im2, F, x1, y1):
    # Replace pass by your implementation


    window = 10
    sigma = 3
    err_min = float('inf')

    weights = g_kernel(window*2+1, sigma)
    if len(im1.shape) > 2:
        weights = np.repeat(weights[:, :, None], im1.shape[-1], axis=2)

    # epipolar line,
    l = F @ np.array([[x1], [y1], [1]])

    # ax+by+c = 0
    a = l[0]
    b = l[1]
    c = l[2]

    y1 = int(y1)
    x1 = int(x1)

    # suppose the window in both im1 and im2 does not meet the boundary
    im1_window = im2[y1-window: y1+window+1, x1-window:x1+window+1, :]
    for y in range(window, im2.shape[0] - window):
        x = int( ( -c - b*y * 1.0 ) / a )
        # x = ( ( -c - b*y * 1.0 ) / a )
        # print(int(x))

        if x-window >= 0 and x+window < im2.shape[1]:
            im2_window = im2[y-window: y+window+1, x-window:x+window+1, :]

            dist = im2_window - im1_window

            # err = gaussian_filter(dist, sigma=sigma)
            # err = np.sum(np.square(err))

            err = np.sum(np.square(dist) * weights)
            

            if err < err_min:
                x2 = x
                y2 = y
                err_min = err
    
    return x2, y2



'''
Q3.5.1: RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers
'''
def ransacF(pts1, pts2, M):
    # Replace pass by your implementation
    iters = int(1e2)
    thr = 1
    max_inliers = 0

    N = pts1.shape[0]

    pts1_homo = np.hstack((pts1, np.ones((N, 1))))
    pts2_homo = np.hstack((pts2, np.ones((N, 1))))

    for i in range(iters):
        idx = np.random.choice(N, 8, replace=False)
        F_temp = eightpoint(pts1[idx,:], pts2[idx,:], M)

        # obj = pts1_homo @ F_temp @ pts2_homo.T
        # inliers_temp = obj.diagonal() < thr

        # distance from point to epipolar line.
        # l' = F x
        l = F_temp @ pts1_homo.T
        # d = dist(x', l')
        # d = ?

        d = abs(pts2_homo @ l)
        a2 = np.square(l[0,:])
        b2 = np.square(l[1,:])
        d /= np.sqrt(a2[:, None] + b2[:, None])
        inliers_temp = d.diagonal() < thr

        if np.sum(inliers_temp) > max_inliers:
            F = F_temp
            inliers = inliers_temp
            max_inliers = np.sum(inliers_temp)

            print('number of inliers:', max_inliers)
            if max_inliers == N:
                break
        
        print('numer of interations:', i)
        print('current max_inliers:', max_inliers)

    # F = eightpoint(pts1[inliers, :], pts2[inliers, :], M)
    return F, inliers




'''
Q3.5.2: Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
# Ref: https://mathworld.wolfram.com/RodriguesRotationFormula.html

def rodrigues(r):
    # Replace pass by your implementation
    theta = np.linalg.norm(r)

    if theta == 0:
        return np.eye(3)
    else:
        w = r/theta
        K = np.array([
            [0,      -w[2],  w[1]],
            [w[2],   0,      -w[0]],
            [-w[1],  w[0],   0]
        ])
        return np.eye(3) + np.sin(theta) * K + (1-np.cos(theta))* K @ K

'''
Q3.5.2: Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''

def invRodrigues(R):
    # Replace pass by your implementation
    thr = 1e-8
    cos_theta = (np.sum(R.diagonal()) - 1) / 2.0
    theta = np.arccos(cos_theta)
    
    r = np.zeros(3)

    if abs(theta) > thr:
        r[0] = np.sqrt( (R[0,0] - cos_theta ) / (1-cos_theta) )
        r[1] = np.sqrt( (R[1,1] - cos_theta ) / (1-cos_theta) )
        r[2] = np.sqrt( (R[2,2] - cos_theta ) / (1-cos_theta) )

    return r


'''
Q3.5.3: Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original 
            and estimated projections
'''
def rodriguesResidual(K1, M1, p1, K2, p2, x):
    # Replace pass by your implementation
    w, r2, t2 = x[:-6], x[-6:-3], x[-3:]
    # print(r2.shape)
    w = w.reshape(-1, 3)
    # print(w.shape)
    R2 = rodrigues(r2)
    M2 = np.hstack((R2, t2[:, None]))
    # print(M2)

    # camera matrix
    C1 = K1 @ M1
    C2 = K2 @ M2
    homo_w = np.hstack((w, np.zeros((w.shape[0], 1))))

    # projections
    hat_p1 = homo_w @ C1.T
    hat_p2 = homo_w @ C2.T

    hat_p1 = hat_p1[:, 0:2] / hat_p1[:, -1][:, None]
    hat_p2 = hat_p2[:, 0:2] / hat_p2[:, -1][:, None]

    res = np.concatenate([ (p1-hat_p1).reshape([-1]), (p2-hat_p2).reshape([-1]) ])
    return res
    # return res[:, None]


'''
Q3.5.3 Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
'''

def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    # Replace pass by your implementation
    R2 = M2_init[:, 0:3]
    r2 = invRodrigues(R2)

    t2 = M2_init[:, -1]
    x = np.concatenate([P_init.reshape([-1]), r2, t2])


    func = lambda x: rodriguesResidual(K1, M1, p1, K2, p2, x)
    # x = x[:,]
    # y = func(x)
    x, _ = leastsq(func, x)

    w, r2, t2 = x[:-6], x[-6:-3] ,x[-3:]
    P2 = w.reshape(-1, 3)

    R2 = rodrigues(r2)
    M2 = np.hstack((R2, t2[:, None]))

    return M2, P2



def renderNDotLSphere(center, rad, light, pxSize, res):

    """
    Q4.1

    Render a sphere with a given center and radius. The camera is 
    orthographic and looks towards the sphere in the negative z
    direction. The camera's sensor axes are centerd on and aligned
    with the x- and y-axes.

    Parameters
    ----------
    center : numpy.ndarray
        The center of the hemispherical bowl in an array of size (3,)

    rad : float
        The radius of the bowl

    light : numpy.ndarray
        The direction of incoming light

    pxSize : float
        Pixel size

    res : numpy.ndarray
        The resolution of the camera frame

    Returns
    -------
    image : numpy.ndarray
        The rendered image of the hemispherical bowl
    """
    # Replace pass by your implementation
    rad = rad // pxSize
    center = center // pxSize

    x = np.arange(res[0])
    y = np.arange(res[1])
    [x, y] = np.meshgrid(x, y)

    # transform the coordinates to center  of the image
    xx = x - res[0]//2 + center[0]
    yy = y - res[1]//2 + center[1]
    z2 =  rad**2 - xx**2 - yy**2

    # for mask < 0
    idx = z2 < 0
    z2[idx] = 0
    zz = np.sqrt(z2)

    # normal, [x/z y/z 1]
    normal_z = np.where(zz == 0, 1, zz)

    normal_x = (x-res[0]//2)/normal_z
    normal_y = (y-res[1]//2)/normal_z


    # x,y --> x, -y
    intensity = (normal_x*light[0] - normal_y*light[1] + light[2] ) / np.sqrt(1+normal_x**2+normal_y**2)
    intensity = np.where(intensity < 0, 0, intensity)
    intensity[idx] = 0
    return intensity




def loadData(path = "../data/"):

    """
    Q4.2.1

    Load data from the path given. The images are stored as input_n.tif
    for n = {1...7}. The source lighting directions are stored in
    sources.mat.

    Paramters
    ---------
    path: str
        Path of the data directory

    Returns
    -------
    I : numpy.ndarray
        The 7 x P matrix of vectorized images

    L : numpy.ndarray
        The 3 x 7 matrix of lighting directions

    s: tuple
        Image shape

    """
    # Replace pass by your implementation
    n = 7
    I = []
    L = []

    for i in range(n):
        im = plt.imread(path + 'input_' + str(i+1) + '.tif')
        im_Y = utils.lRGB2XYZ(im)[:,:,1]
        I.append(im_Y.reshape(-1))
    I = np.stack(I, axis=0)
    s = im.shape

    L = np.load(path + 'sources.npy').T
    return I, L, s


def estimatePseudonormalsCalibrated(I, L):

    """
    Q4.2.2

    In calibrated photometric stereo, estimate pseudonormals from the
    light direction and image matrices

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P array of vectorized images

    L : numpy.ndarray
        The 3 x 7 array of lighting directions

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pesudonormals
    """
    # Replace pass by your implementation
    # B = lsqr(L.T, I)[0]
    B = np.linalg.lstsq(L.T, I)[0]
    return B


def estimateAlbedosNormals(B):

    '''
    Q4.2.3

    From the estimated pseudonormals, estimate the albedos and normals

    Parameters
    ----------
    B : numpy.ndarray
        The 3 x P matrix of estimated pseudonormals

    Returns
    -------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals
    '''
    # Replace pass by your implementation
    albedos = np.linalg.norm(B, axis=0)
    normals = B / albedos
    return albedos, normals


def displayAlbedosNormals(albedos, normals, s):

    """
    Q4.2.4

    From the estimated pseudonormals, display the albedo and normal maps

    Please make sure to use the `coolwarm` colormap for the albedo image
    and the `rainbow` colormap for the normals.

    Parameters
    ----------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    -------
    albedoIm : numpy.ndarray
        Albedo image of shape s

    normalIm : numpy.ndarray
        Normals reshaped as an s x 3 image

    """
    # Replace pass by your implementation
    albedoIm = albedos.reshape(s[:2])
    plt.imshow(albedoIm, cmap='gray')
    plt.show()

    normals += abs(np.min(normals))
    normals /= np.max(normals)
    
    im = []
    for i in range(3):
        im.append(normals[i,:].reshape(s[:2]))
    im = np.stack(im, axis=-1)
    plt.imshow(im, cmap='rainbow')
    plt.show()




def estimateShape(normals, s):

    """
    Q4.3.1

    Integrate the estimated normals to get an estimate of the depth map
    of the surface.

    Parameters
    ----------
    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    ----------
    surface: numpy.ndarray
        The image, of size s, of estimated depths at each point

    """
    # Replace pass by your implementation
    pass


def plotSurface(surface):

    """
    Q4.3.1

    Plot the depth map as a surface

    Parameters
    ----------
    surface : numpy.ndarray
        The depth map to be plotted

    Returns
    -------
        None

    """
    # Replace pass by your implementation
    pass