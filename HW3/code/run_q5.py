import numpy as np
import scipy.io
from nn import *
from collections import Counter
import matplotlib.pyplot as plt

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

max_iters = 100 + 1
# pick a batch size, learning rate
batch_size = 36 
learning_rate =  3e-5
hidden_size = 32
lr_rate = 20
batches = get_random_batches(train_x,np.ones((train_x.shape[0],1)),batch_size)
batch_num = len(batches)

momentum = 0.9

params = Counter()

# Q5.1 & Q5.2
# initialize layers here
##########################
##### your code here #####
##########################

initialize_weights(1024, 32, params, 'layer1')
initialize_weights(32, 32, params, 'layer2')
initialize_weights(32, 32, params, 'layer3')
initialize_weights(32, 1024, params, 'output')

train_loss = []
# should look like your previous training loops
for itr in range(max_iters):
    total_loss = 0
    for xb,_ in batches:
        # training loop can be exactly the same as q2!
        # your loss is now squared error
        # delta is the d/dx of (x-y)^2
        # to implement momentum
        #   just use 'm_'+name variables
        #   to keep a saved value over timestamps
        #   params is a Counter(), which returns a 0 if an element is missing
        #   so you should be able to write your loop without any special conditions

        ##########################
        ##### your code here #####
        ##########################

        # forward
        h = forward(xb, params, 'layer1', relu)
        h = forward(h, params, 'layer2', relu)
        h = forward(h, params, 'layer3', relu)
        out = forward(h, params, 'output', sigmoid)

        # loss
        total_loss += np.sum(np.square(xb - out))

        # backwards
        delta = -2 * (xb - out)
        delta = backwards(delta, params, 'output', sigmoid_deriv)
        delta = backwards(delta, params, 'layer3', relu_deriv)
        delta = backwards(delta, params, 'layer2', relu_deriv)
        delta = backwards(delta, params, 'layer1', relu_deriv)

        # apply gradient
        # momentum
        params['m_blayer1'] = momentum * params['m_blayer1'] - learning_rate * params['grad_blayer1']
        params['m_Wlayer1'] = momentum * params['m_Wlayer1'] - learning_rate * params['grad_Wlayer1']

        params['m_blayer2'] = momentum * params['m_blayer2'] - learning_rate * params['grad_blayer2']
        params['m_Wlayer2'] = momentum * params['m_Wlayer2'] - learning_rate * params['grad_Wlayer2']

        params['m_blayer3'] = momentum * params['m_blayer3'] - learning_rate * params['grad_blayer3']
        params['m_Wlayer3'] = momentum * params['m_Wlayer3'] - learning_rate * params['grad_Wlayer3']

        params['m_boutput'] = momentum * params['m_boutput'] - learning_rate * params['grad_boutput']
        params['m_Woutput'] = momentum * params['m_Woutput'] - learning_rate * params['grad_Woutput']        

        # update
        params['blayer1'] += params['m_blayer1']
        params['Wlayer1'] += params['m_Wlayer1']

        params['blayer2'] += params['m_blayer2']
        params['Wlayer2'] += params['m_Wlayer2']

        params['blayer3'] += params['m_blayer3']
        params['Wlayer3'] += params['m_Wlayer3']

        params['boutput'] += params['m_boutput']
        params['Woutput'] += params['m_Woutput']

    train_loss.append(total_loss)
    

    if itr % 5 == 0:
        print("itr: {:02d} \t loss: {:.2f}".format(itr,total_loss))
    if itr % lr_rate == lr_rate-1:
        learning_rate *= 0.9


# Q5.2 Plots
if False:
    plt.plot(train_loss, 'b')
    plt.xlabel('epoch')
    plt.ylabel('loss')

    plt.show()

# Q5.3.1

# visualize some results
##########################
##### your code here #####
##########################
from mpl_toolkits.axes_grid1 import ImageGrid

h = forward(valid_x, params, 'layer1', relu)
h = forward(h, params, 'layer2', relu)
h = forward(h, params, 'layer3', relu)
out = forward(h, params, 'output', sigmoid)

if False:
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

    vis = out[indices].reshape(num, 32, 32)

    displayed = np.zeros((num*2, 32, 32))
    displayed[::2] = images
    displayed[1::2] = vis
    for ax, im in zip(grid, displayed):
        ax.imshow(im.T)
    plt.savefig("out.jpg")
    plt.show()


# Q5.3.2
from skimage.measure import compare_psnr as psnr
# evaluate PSNR
##########################
##### your code here #####
##########################
# valid_x = valid_x.astype(np.float)
avg_psnr = np.mean([ psnr(valid_x[i], out[i]) for i in range(valid_x.shape[0]) ])
print(avg_psnr)



