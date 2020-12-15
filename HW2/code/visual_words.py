import numpy as np
import multiprocessing
import scipy.ndimage
import skimage
import sklearn.cluster
import scipy.spatial.distance
import os, time
import matplotlib.pyplot as plt
import util
import cv2
import skimage.filters

def extract_filter_responses(image):
    '''
    Extracts the filter responses for the given image.

    [input]
    * image: numpy.ndarray of shape (H, W) or (H, W, 3)

    [output]
    * filter_responses: numpy.ndarray of shape (H, W, 3F)
    '''

    # ----- TODO -----
    if len(image.shape) == 2:
        image = np.dstack([image, image, image])

    image = skimage.color.rgb2lab(image)
    img_out = []

    scale_sets = [1, 2, 4, 8, 8*np.sqrt(2)]
    for scale in scale_sets:
        for img in (image[:,:,0], image[:,:,1], image[:,:,2]):
            # print(scale)
            # print(img.shape)

            img_out.append(scipy.ndimage.gaussian_filter(img, sigma=scale, mode='nearest'))
            img_out.append(scipy.ndimage.gaussian_laplace(img, sigma=scale, mode='nearest'))

            img_out.append(scipy.ndimage.gaussian_filter(img, sigma=scale, order=[1,0], mode='nearest' ))
            img_out.append(scipy.ndimage.gaussian_filter(img, sigma=scale, order=[0,1], mode='nearest'))

    return np.dstack(img_out)

def get_visual_words(image, dictionary):
    '''
    Compute visual words mapping for the given image using the dictionary of visual words.

    [input]
    * image: numpy.ndarray of shape (H, W) or (H, W, 3)

    [output]
    * wordmap: numpy.ndarray of shape (H, W)
    '''

    # ----- TODO -----
    image = extract_filter_responses(image)
    h, w = image.shape[0], image.shape[1]
    image = image.reshape(-1, image.shape[2])

    d = scipy.spatial.distance.cdist(image, dictionary, metric='euclidean')
    idx = np.argmin(d, axis=1)
    wordmap = idx.reshape(h, w)
    return wordmap


def get_harris_points(image, alpha, k = 0.05):
    '''
    Compute points of interest using the Harris corner detector

    [input]
    * image: numpy.ndarray of shape (H, W) or (H, W, 3)
    * alpha: number of points of interest desired
    * k: senstivity factor 

    [output]
    * points_of_interest: numpy.ndarray of shape (alpha, 2) that contains interest points
    '''

    # ----- TODO -----
    
    if len(image.shape) == 3:
        image = skimage.color.rgb2gray(image)

    # print(image.shape)

    # p = 3 or 5
    p = 3

    # why not working?
    # Ix = cv2.Sobel(image, cv2.CV_32F, 1, 0)
    # Iy = cv2.Sobel(image, cv2.CV_32F, 0, 1)
    Ix = skimage.filters.sobel(image, axis=1)
    Iy = skimage.filters.sobel(image, axis=0)

    Ixx = np.multiply(Ix, Ix)
    Ixy = np.multiply(Ix, Iy)
    Iyy = np.multiply(Iy, Iy)
    
    filter_ones = np.ones((p, p))
    Ixx = scipy.ndimage.convolve(Ixx, filter_ones)
    Ixy = scipy.ndimage.convolve(Ixy, filter_ones)
    Iyy = scipy.ndimage.convolve(Iyy, filter_ones)

    # R is a matrix
    R = np.multiply(Ixx,Iyy) - np.multiply(Ixy,Ixy) - k * np.square(Ixx + Iyy)

    idx = np.argsort(R.flatten())[-alpha:]
    idx = np.unravel_index(idx, image.shape)
    
    # print(idx)
    # print(np.array([idx[0], idx[1]]).T.shape)
    return np.array([idx[0], idx[1]]).T



def compute_dictionary_one_image(args):
    '''
    Extracts alpha samples of the dictionary entries from an image. Use the 
    harris corner detector implmented from previous question to extract 
    the point of interests. This should be a function run by a subprocess.

    [input]
    * i: index of training image
    * alpha: number of samples
    * image_path: path of image file

    [saved]
    * sampled_response: numpy.ndarray of shape (alpha, 3F)
    '''


    i, alpha, image_path = args
    # ----- TODO -----
    path_img = '../data/' + image_path
    image = skimage.io.imread(path_img)
    image = image.astype('float')/255

    # * filter_responses: numpy.ndarray of shape (H, W, 3F)
    filter_responses = extract_filter_responses(image)
    # * points_of_interest: numpy.ndarray of shape (alpha, 2)
    points_of_interest = get_harris_points(image, alpha)

    y, x = points_of_interest[:, 1], points_of_interest[:, 0]
    Sampled_response = filter_responses[x, y, :]
    # print('filter:', filter_responses.shape)
    # print('sampled:', Sampled_response.shape)
    # Sampled_response = Sampled_response.reshape[-1, Sampled_response.shape[2]]

    np.save('../data/Q1_train/Sample{}.npy'.format(i), Sampled_response)
    
    print('Iteration of images:', i)


def compute_dictionary(num_workers=2):
    '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * num_workers: number of workers to process in parallel

    [saved]
    * dictionary: numpy.ndarray of shape (K, 3F)
    '''
    # get names of iamges, train_data['files']
    train_data = np.load("../data/train_data.npz")

    # ----- TODO -----
    train_data_names = train_data['files']
    # K between 100 and 300
    K = 100
    # alpha between 50 and 500
    alpha = 200
    args = [( i, alpha, train_data_names[i]) for i in range(len(train_data_names)) ]

    # Done!
    # with multiprocessing.Pool(processes=num_workers) as pool:
    #     pool.map(compute_dictionary_one_image, args)
    
    print('Done with one image process.')
    responses = []
    for name in os.listdir('../data/Q1_train/'):
        responses.append(np.load('../data/Q1_train/'+name))
    responses = np.vstack(responses)

    print('Start K-means:')
    kmeans = sklearn.cluster.KMeans(n_clusters=K).fit(responses)
    dictionary = kmeans.cluster_centers_

    np.save('dictionary.npy', dictionary)
    return dictionary



    


