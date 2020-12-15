import numpy as np
import skimage
import multiprocessing
import threading
import queue
import os,time
import math
import visual_words

def compute_features_one_image(args):
    dictionary = np.load("dictionary.npy")
    SPM_layer_num = 3

    i, name, label = args

    path_img = '../data/' + name
    image = skimage.io.imread(path_img)
    image = image.astype('float')/255

    wordmap = visual_words.get_visual_words(image, dictionary)
    hist_feature = get_feature_from_wordmap_SPM(wordmap, SPM_layer_num, len(dictionary))

    np.savez('../data/Q2_train/Sample{}.npy'.format(i), hist_feature=hist_feature, label=label)
    print('Iteration of images:', i)


def build_recognition_system(num_workers=2):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * num_workers: number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N, M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K, 3F)
    * SPM_layer_num: number of spatial pyramid layers
    '''

    train_data = np.load("../data/train_data.npz")
    dictionary = np.load("dictionary.npy")
    # ----- TODO -----
    names, labels = train_data['files'], train_data['labels']
    args = [(i, names[i], labels[i]) for i in range(len(names))]

    # Done!
    # with multiprocessing.Pool(processes=num_workers) as pool:
    #     pool.map(compute_features_one_image, args)

    print('Start reading...')
    features = []
    labels = []
    for name in os.listdir('../data/Q2_train/'):
        temp = np.load('../data/Q2_train/'+name)
        features.append(temp['hist_feature'])
        labels.append(temp['label'])
    
    features = np.asarray(features)
    labels = np.asarray(labels)
    np.savez('trained_system.npz', dictionary=dictionary, features=features, labels=labels, SPM_layer_num=3)


# for parallel but not implemented yet
def evaluate_one_image(args):
    dictionary = np.load("dictionary.npy")
    trained_system = np.load("trained_system.npz")
    SPM_layer_num = 3

    train_features, train_labels = trained_system['features'], trained_system['labels']
    trained_system = np.load("trained_system.npz")

    i, name, true_label = args

    path_img = '../data/' + name
    image = skimage.io.imread(path_img)
    image = image.astype('float')/255

    wordmap = visual_words.get_visual_words(image, dictionary)
    feature = get_feature_from_wordmap_SPM(wordmap, SPM_layer_num, len(dictionary))

    idx = np.argmax(distance_to_set(feature, train_features))
    pred_label = int(train_labels[idx])
    # true_labels = int(test_labels[i])

    np.savez('../data/Q2_test/Sample{}.npy'.format(i), pred_label=pred_label, true_label=true_label)
    print('Iteration of images:', i)


def evaluate_recognition_system(num_workers=2):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * num_workers: number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8, 8)
    * accuracy: accuracy of the evaluated system
    '''

    test_data = np.load("../data/test_data.npz")
    trained_system = np.load("trained_system.npz")
    dictionary = np.load("dictionary.npy")
    SPM_layer_num = 3
    # ----- TODO -----
    train_features, train_labels = trained_system['features'], trained_system['labels']
    names, test_labels = test_data['files'], test_data['labels']
    path = '../data/'

    conf = np.zeros((8, 8))
    for i in range(len(names)):
        image = skimage.io.imread(path + names[i])
        image = image.astype('float')/255

        wordmap = visual_words.get_visual_words(image, dictionary)
        feature = get_feature_from_wordmap_SPM(wordmap, SPM_layer_num, len(dictionary))

        idx = np.argmax(distance_to_set(feature, train_features))
        conf[int(test_labels[i])][int(train_labels[idx])] += 1

        if i % 10 == 0:
            print('iteration of test images: {}'.format(i))
    
    accuracy = np.trace(conf)/len(names)
    return conf, accuracy





def get_image_feature(file_path, dictionary, layer_num, K):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * file_path: path of image file to read
    * dictionary: numpy.ndarray of shape (K, 3F)
    * layer_num: number of spatial pyramid layers
    * K: number of clusters for the word maps

    [output]
    * feature: numpy.ndarray of shape (K*(4^layer_num-1)/3)
    '''
    # ----- TODO -----

    pass


def distance_to_set(word_hist, histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N, K)

    [output]
    * sim: numpy.ndarray of shape (N)
    '''
    # ----- TODO -----

    return np.sum(np.minimum(word_hist, histograms), axis=1)


def get_feature_from_wordmap(wordmap, dict_size):
    '''
    Compute histogram of visual words.

    [input]
    * wordmap: numpy.ndarray of shape (H, W)
    * dict_size: dictionary size K

    [output]
    * hist: numpy.ndarray of shape (K)
    '''

    # ----- TODO -----
    hist, _ = np.histogram(wordmap, bins=np.arange(dict_size+1), density=True)
    return hist


def get_feature_from_wordmap_SPM(wordmap, layer_num, dict_size):
    '''
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * wordmap: numpy.ndarray of shape (H, W)
    * layer_num: number of spatial pyramid layers
    * dict_size: dictionary size K

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^layer_num-1)/3)
    '''

    # ----- TODO -----
    
    L = layer_num - 1
    hist_all = []

    # finest block
    n = 2**L

    # when L = 2, w = 1/2
    for row_block in np.array_split(wordmap, n, axis=0):
        for block in np.array_split(row_block, n, axis=1):
            # print(block.shape)
            temp = get_feature_from_wordmap(block, dict_size)
            hist_all.append( temp / 2.0) 

    # L = 1, w = 1/4
    for idx in [0, 2, 8, 10]:
        temp = hist_all[idx] + hist_all[idx+1] + hist_all[idx+4] + hist_all[idx+5]
        hist_all.append( temp/4. )


    # L = 0,  w = 1/4
    temp = 0
    for i in range(n):
        temp += hist_all[i]
    hist_all.append(temp / 4.)


    hist_all = np.asarray(hist_all)
    hist_all = hist_all.reshape(-1)
    return hist_all / np.sum(hist_all)



    