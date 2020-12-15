import numpy as np
import scipy.io
from nn import *
import matplotlib.pyplot as plt

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']

max_iters = 51
# pick a batch size, learning rate
batch_size = 16
learning_rate = 5e-3
hidden_size = 64
##########################
##### your code here #####
##########################

batches = get_random_batches(train_x, train_y, batch_size)
batch_num = len(batches)

params = {}

# initialize layers here
##########################
##### your code here #####
##########################
N, D = train_x.shape
_, num_classes = train_y.shape

# initialize a layer
initialize_weights(D, hidden_size, params, 'layer1')
initialize_weights(hidden_size, num_classes, params,'output')


train_loss = []
valid_loss = []

train_acc = []
valid_acc = []

# with default settings, you should get loss < 150 and accuracy > 80%
for itr in range(max_iters):
    total_loss = 0
    total_acc = 0
    for xb,yb in batches:
        # training loop can be exactly the same as q2!
        ##########################
        ##### your code here #####
        ##########################

        # forward
        h1 = forward(xb,params,'layer1')
        probs = forward(h1,params,'output',softmax)

        # loss
        # be sure to add loss and accuracy to epoch totals 
        loss, acc = compute_loss_and_acc(yb, probs)
        total_loss += loss
        total_acc += acc

        # backward
        delta = probs - yb
        delta = backwards(delta, params,'output', linear_deriv)
        backwards(delta, params, 'layer1', sigmoid_deriv)

        # apply gradient
        params['blayer1'] -= learning_rate * params['grad_blayer1']
        params['Wlayer1'] -= learning_rate * params['grad_Wlayer1']

        params['boutput'] -= learning_rate * params['grad_boutput']
        params['Woutput'] -= learning_rate * params['grad_Woutput']



    avg_acc = total_acc / batch_num
    avg_loss = total_loss / N

    train_loss.append(avg_loss)
    train_acc.append(avg_acc)
    if itr % 5 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr, avg_loss, avg_acc))

    # run on validation set and report accuracy! should be above 75%
    # valid_acc = None
    ##########################
    ##### your code here #####
    ##########################

    # forward
    h1 = forward(valid_x, params,'layer1')
    probs = forward(h1, params,'output', softmax)

    # loss
    loss, acc = compute_loss_and_acc(valid_y, probs)
    loss /= valid_x.shape[0]

    valid_loss.append(loss)
    valid_acc.append(acc)
    if itr % 5 == 0:
        print("itr: {:02d} \t valid_loss: {:.2f} \t valid_acc : {:.2f}".format(itr, loss, acc))



# print('Validation accuracy: ', valid_acc)
if False: # view the data
    # print('view the data: ')
    for crop in xb:
        import matplotlib.pyplot as plt
        plt.imshow(crop.reshape(32,32).T)
        plt.show()
import pickle
saved_params = {k:v for k,v in params.items() if '_' not in k}
with open('q3_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Q3.1 Plots
if False:
    plt.subplot(1,2,1)
    plt.plot(train_acc, 'b', label = 'train')
    plt.plot(valid_acc, 'r', label = 'valid')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(train_loss, 'b', label = 'train')
    plt.plot(valid_loss, 'r', label = 'valid')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()

    plt.show()



# # Q3.1.3
# import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

# fig = plt.figure(1, (8., 8.))
# if hidden_size < 128:
#     grid = ImageGrid(fig, 111,  # similar to subplot(111)
#                     nrows_ncols=(8, 8),  # creates 2x2 grid of axes
#                     axes_pad=0.1,  # pad between axes in inch.
#                     )
#     img_w = params['Wlayer1'].reshape((32,32,hidden_size))
#     for i in range(hidden_size):
#         grid[i].imshow(img_w[:,:,i])  # The AxesGrid object work as a list of axes.

#     plt.show()


# # Q3.1.4
# fig = plt.figure(1, (6., 8.))
# grid = ImageGrid(fig, 111,  # similar to subplot(111)
#                  nrows_ncols=(12, 6),  # creates 2x2 grid of axes
#                  axes_pad=0.1,  # pad between axes in inch.
#                  )

# indices = params['cache_output'][2].argmax(axis=0)
# images = valid_x[indices]
# images = images.reshape(36, 32, 32)

# vis = np.zeros((36, 1024))
# inps = np.eye(36)
# for i,inp in enumerate(inps):
#     vis[i] = inp @ params['Woutput'].T @ params['Wlayer1'].T 
# vis = vis.reshape(36, 32, 32)

# displayed = np.zeros((72, 32, 32))
# displayed[::2] = images
# displayed[1::2] = vis
# for ax, im in zip(grid, displayed):
#     ax.imshow(im.T)
# plt.savefig("out.jpg")
# plt.show()

# Q3.1.5
confusion_matrix = np.zeros((train_y.shape[1],train_y.shape[1]))

# compute comfusion matrix here
##########################
##### your code here #####
##########################
train_data = scipy.io.loadmat('../data/nist36_test.mat')
test_x, test_y = train_data['test_data'], train_data['test_labels']

h1 = forward(test_x, params,'layer1')
probs = forward(h1, params,'output',softmax)
true_label = np.argmax(test_y, axis=1)
pred_label = np.argmax(probs, axis=1)

for i in range(len(true_label)):
    confusion_matrix[true_label[i]][pred_label[i]] += 1

import string
plt.imshow(confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.show()