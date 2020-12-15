import numpy as np
import scipy.io
from nn import *
import matplotlib.pyplot as plt
from skimage.measure import compare_psnr as psnr
from mpl_toolkits.axes_grid1 import ImageGrid

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

dim = 32
# do PCA
##########################
##### your code here #####
##########################

# standardization, independently on each feature
def standardization(x):
    mu = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    return (x-mu)/std

train_x = standardization(train_x)
C_train_x = np.dot(train_x.T, train_x) / train_x.shape[0]
U, S, V = np.linalg.svd(C_train_x)

proj = U[:,:dim]


# rebuild a low-rank version
lrank = None
##########################
##### your code here #####
##########################
lrank = np.dot(train_x, proj)


# rebuild it
recon = None
##########################
##### your code here #####
##########################
recon = np.dot(lrank, proj.T)

# build valid dataset
recon_valid = None
##########################
##### your code here #####
##########################
# valid_x = standardization(valid_x)
recon_valid = standardization(valid_x).dot(proj).dot(proj.T)

mu = np.mean(valid_x, axis=0)
std = np.std(valid_x, axis=0)
recon_valid = recon_valid * std + mu

# visualize the comparison and compute PSNR
##########################
##### your code here #####
##########################
if True:
    # fig = plt.figure(1, (6., 8.))
    fig = plt.figure()
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(2, 10),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )
    num = 10
    indices = [0,1, 500,501, 1000,1001, 1500,1501, 2000,2001]
    images = valid_x[indices]
    images = images.reshape(num, 32, 32)

    vis = recon_valid[indices].reshape(num, 32, 32)

    displayed = np.zeros((num*2, 32, 32))
    displayed[::2] = images
    displayed[1::2] = vis
    for ax, im in zip(grid, displayed):
        ax.imshow(im.T)
    # plt.savefig("out.jpg")
    plt.show()


# evaluate PSNR
##########################
##### your code here #####
##########################
# valid_x = valid_x.astype(np.float)
# mu = np.mean(valid_x, axis=0)
# std = np.std(valid_x, axis=0)
# recon_valid = recon_valid * std + mu
avg_psnr = np.mean([ psnr(valid_x[i], recon_valid[i]) for i in range(valid_x.shape[0]) ])
print(avg_psnr)