import numpy as np
import multiprocessing
import threading
import queue
import os,time
import torch
import skimage.transform
import torchvision.transforms
import util
import network_layers
from skimage.transform import resize

torch.set_num_threads(1)  # without this line, pytroch forward pass will hang with multiprocessing

def evaluate_deep_extractor(img, vgg16):
    '''
    Evaluates the deep feature extractor for a single image.

    [input]
    * image: numpy.ndarray of shape (H,W,3)
    * vgg16: prebuilt VGG-16 network.

    [output]
    * diff: difference between the two feature extractor's result
    '''
    vgg16_weights = util.get_VGG16_weights()
    img_torch = preprocess_image(img)
    
    feat = network_layers.extract_deep_feature(np.transpose(img_torch.numpy(), (1,2,0)), vgg16_weights)
    
    with torch.no_grad():
        vgg_classifier = torch.nn.Sequential(*list(vgg16.classifier.children())[:-3])
        vgg_feat_feat = vgg16.features(img_torch[None, ])
        vgg_feat_feat = vgg_classifier(vgg_feat_feat.flatten())
    
    return np.sum(np.abs(vgg_feat_feat.numpy() - feat))


def build_recognition_system(vgg16, num_workers=2):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * vgg16: prebuilt VGG-16 network.
    * num_workers: number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N, K)
    * labels: numpy.ndarray of shape (N)
    '''

    train_data = np.load("../data/train_data.npz")

    # ----- TODO -----
    
    names, labels = train_data['files'], train_data['labels']
    args = [(i, names[i], labels[i], vgg16) for i in range(len(names))]

    # # Done!
    # with multiprocessing.Pool(processes=num_workers) as pool:
    #     pool.map(get_image_feature, args)

    print('Start reading...')
    features = []
    labels = []
    for name in os.listdir('../data/Q4_train/'):
        temp = np.load('../data/Q4_train/'+name)
        features.append(temp['feature'])
        labels.append(temp['label'])
    
    features = np.asarray(features)
    labels = np.asarray(labels)
    np.savez('trained_system_deep.npz', features=features, labels=labels)
    

def evaluate_recognition_system(vgg16, num_workers=2):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * vgg16: prebuilt VGG-16 network.
    * num_workers: number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8, 8)
    * accuracy: accuracy of the evaluated system
    '''

    test_data = np.load("../data/test_data.npz")
    trained_system = np.load("trained_system_deep.npz")

    # ----- TODO -----
    
    train_features, train_labels = trained_system['features'], trained_system['labels']
    names, test_labels = test_data['files'], test_data['labels']
    path = '../data/'

    conf = np.zeros((8, 8))
    for i in range(len(names)):
        image = skimage.io.imread(path + names[i])
        image = image.astype('float')/255

        img_torch = preprocess_image(image)

        with torch.no_grad():
            vgg_classifier = torch.nn.Sequential(*list(vgg16.classifier.children())[:-3])
            vgg_feat_feat = vgg16.features(img_torch[None, ])
            vgg_feat_feat = vgg_classifier(vgg_feat_feat.flatten())

            idx = np.argmin(distance_to_set(vgg_feat_feat, train_features))
            conf[int(test_labels[i])][int(train_labels[idx])] += 1

            if i % 10 == 0:
                print('iteration of test images: {}'.format(i))
    
    accuracy = np.trace(conf)/len(names)
    return conf, accuracy


def preprocess_image(image):
    '''
    Preprocesses the image to load into the prebuilt network.

    [input]
    * image: numpy.ndarray of shape (H, W, 3)

    [output]
    * image_processed: torch.Tensor of shape (3, H, W)
    '''

    # ----- TODO -----
    if len(image.shape) == 2:
        image = np.dstack([image, image, image])

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    image = resize(image, (224,224,3))

    for i in range(3):
        image[:,:,i] = (image[:,:,i] - mean[i]) / std[i]
    image = np.transpose(image, (2, 0, 1))
    image = torch.from_numpy(image).type(torch.DoubleTensor) 

    return image




def get_image_feature(args):
    '''
    Extracts deep features from the prebuilt VGG-16 network.
    This is a function run by a subprocess.
    [input]
    * i: index of training image
    * image_path: path of image file
    * vgg16: prebuilt VGG-16 network.
    
    [output]
    * feat: evaluated deep feature
    '''

    i, name, label, vgg16 = args

    # ----- TODO -----

    path_img = '../data/' + name
    image = skimage.io.imread(path_img)
    img_torch = preprocess_image(image)

    with torch.no_grad():
        vgg_classifier = torch.nn.Sequential(*list(vgg16.classifier.children())[:-3])
        vgg_feat_feat = vgg16.features(img_torch[None, ])
        vgg_feat_feat = vgg_classifier(vgg_feat_feat.flatten())

    np.savez('../data/Q4_train/Sample{}.npy'.format(i), feature=vgg_feat_feat, label=label)
    print('interation of image: {}'.format(i))




def distance_to_set(feature, train_features):
    '''
    Compute distance between a deep feature with all training image deep features.

    [input]
    * feature: numpy.ndarray of shape (K)
    * train_features: numpy.ndarray of shape (N, K)

    [output]
    * dist: numpy.ndarray of shape (N)
    '''

    # ----- TODO -----
    return np.linalg.norm(feature - train_features, axis=1)