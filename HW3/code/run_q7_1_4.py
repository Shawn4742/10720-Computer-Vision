import torchvision.datasets
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

torchvision.datasets.EMNIST.url = 'http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip'
# emnist = torchvision.datasets.EMNIST(root='../data', split='balanced', download=True)
# print(emnist)

train_data = torchvision.datasets.EMNIST(root='../data', train=True, split='balanced',
                                download=False, transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.EMNIST(root='../data', train=False, split='balanced',
                                download=False, transform=torchvision.transforms.ToTensor())
# print(len(test_data))

max_iters = 0
# pick a batch size, learning rate
batch_size = 32
learning_rate = 1e-3
hidden_size = 64
num_classes = 47
c_size = 6 * 3*3

train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

if False:
# if True:
    for x, y in train_loader:
        # x = x.type(torch.float64)
        x = torch.squeeze(x, 1)
        x = x.numpy()[:6]

        from mpl_toolkits.axes_grid1 import ImageGrid
        fig = plt.figure()
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                        nrows_ncols=(1, 6),  # creates 2x2 grid of axes
                        axes_pad=0.1,  # pad between axes in inch.
                        )
        displayed = x
        for ax, im in zip(grid, displayed):
            ax.imshow(im.T)
        plt.show()
        break

# (18800, 28, 28)
# 28 --> 24 --> 12 --> 6 --> 3
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 6, 2, 2)

        self.fc1 = nn.Linear(c_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)

        # why not view(-1)?
        x = x.view(-1, c_size)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

model = Net()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
# optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)

train_loss = []
train_acc = []

test_loss = []
test_acc = []


# with default settings, you should get loss < 150 and accuracy > 80%
for itr in range(max_iters):
    total_loss = 0
    acc = []
    for xb, yb in train_loader:
        # training loop can be exactly the same as q2!

        # forward
        output = model(xb)

        # acc 
        pred_y = torch.argmax(output, axis=1)
        true_y = yb
        acc.append(torch.sum(pred_y == true_y).item() / xb.size()[0])

        # loss
        # CrossEntropyLoss() does not take one-hot enconding
        loss = torch.nn.CrossEntropyLoss()(output, true_y)
        total_loss += loss.item()

        # backward
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    avg_acc = np.mean(acc)
    avg_loss = loss

    train_loss.append(avg_loss)
    train_acc.append(avg_acc)
    if itr % 5 == 0:
        print("itr: {:02d} \t train_loss: {:.2f} \t train_acc : {:.2f}".format(itr, avg_loss, avg_acc))

    # test
    # forward
    acc = []
    loss = 0
    with torch.no_grad():
        for test_x, test_y in test_loader:
            output = model(test_x)

            # acc 
            pred_y = torch.argmax(output, axis=1)
            true_y = test_y
            acc.append( torch.sum(pred_y == true_y).item() / test_x.size()[0] )
            # loss
            loss += torch.nn.CrossEntropyLoss()(output, true_y).item()

    acc = np.mean(acc)
    test_loss.append(loss)
    test_acc.append(acc)
    if itr % 5 == 0:
        print("itr: {:02d} \t test_loss: {:.2f} \t test_acc : {:.2f}".format(itr, loss, acc))




PATH = '../data/EMNIST/modelpara.pth'
# # save
# torch.save(model.state_dict(), PATH)
# load
model = Net()
model.load_state_dict(torch.load(PATH))

import string
letters = np.array([str(_) for _ in range(10)] + [_ for _ in string.ascii_uppercase[:26]] + [_ for _ in string.ascii_lowercase[:26]])

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
from mpl_toolkits.axes_grid1 import ImageGrid

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation

from nn import *
from q4 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


d = {}
d['01_list.jpg'] = ['TODOLIST', '1MAKEATODOLIST', '2CHECKOFFTHEFIRST', 'THINGONTODOLIST','3REALIZEYOUHAVEALREADY', 
            'COMPLETEED2THINGS', '4REWARDYOURSELFWITH', 'ANAP']
d['02_letters.jpg'] = ['ABCDEFG', 'HIJKLMN', 'OPQRSTU', 'VWXYZ', '1234567890']
d['03_haiku.jpg'] = ['HAIKUSAREEASY', 'BUTSOMETIMESTHEYDONTMAKESENSE', 'REFRIGERATOR']
d['04_deep.jpg'] = ['DEEPLEARNING', 'DEEPERLEARNING', 'DEEPESTLEARNING']
acc = 0
total = 0

for img in os.listdir('../images'):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
    bboxes, bw = findLetters(im1)
    
    # find the rows using..RANSAC, counting, clustering, etc.
    ##########################
    ##### your code here #####
    ##########################

    # sort by rows, the cols
    # minr, minc, maxr, maxc = bbox
    bboxes = sorted(bboxes, key=lambda x: x[0])

    bboxes = np.array(bboxes)
    height = np.max(bboxes[:,2] - bboxes[:,0]) * 0.7
    one_line_boxes = []
    sorted_boxes = []
    minr_prev = np.min(bboxes[:,0])
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox

        if minr - minr_prev > height:
            one_line_boxes = sorted(one_line_boxes, key=lambda x: x[1])
            sorted_boxes.append(one_line_boxes)
            one_line_boxes = [bbox]
        else:
            one_line_boxes.append(bbox)

        minr_prev = minr
    
    sorted_boxes.append(sorted(one_line_boxes, key=lambda x: x[1]))

    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
    ##########################
    ##### your code here #####
    ##########################
    pad_min = 15
    im_sets = []
    
    for one_line in sorted_boxes:
        one_line_im = []
        for box in one_line:
            minr, minc, maxr, maxc = box

            if maxr - minr > maxc - minc:
                pad_r = pad_min
                pad_c = (maxr - minr - (maxc - minc)) // 2 + pad_min
            else:
                pad_c = pad_min
                pad_r = -(maxr - minr - (maxc - minc)) // 2 + pad_min            

            im = bw[minr: maxr+1, minc:maxc+1]
            im = np.pad(im, ((pad_r,pad_r), (pad_c,pad_c)), 'constant', constant_values=(1,1))
            # print('im shape:', im.shape)
            im = skimage.transform.resize(im, (28, 28)).T

            # but why?
            im = skimage.morphology.erosion(im)
            one_line_im.append(im.flatten())

        im_sets.append(one_line_im)
    
    
    ##########################
    ##### your code here #####
    ##########################

    text = d[img]
    count = 0
    
    for one_line in im_sets:
        one_line = np.asarray(one_line)
        with torch.no_grad():
            x = one_line.reshape(one_line.shape[0], 1, 28, 28)
            x = 1-x

            x = torch.from_numpy(x)
            # x = torch.unsqueeze(x, 1)
            x = x.type(torch.float32)

            if False:
            # if True:
                x = torch.squeeze(x, 1)
                x = x.numpy()[:6]

                from mpl_toolkits.axes_grid1 import ImageGrid
                fig = plt.figure()
                grid = ImageGrid(fig, 111,  # similar to subplot(111)
                                nrows_ncols=(1, 6),  # creates 2x2 grid of axes
                                axes_pad=0.1,  # pad between axes in inch.
                                )
                displayed = x
                for ax, im in zip(grid, displayed):
                    ax.imshow(im.T)
                plt.show()


            output = model(x)

        pred_label = torch.argmax(output, axis=1).tolist()

        for i in range(len(pred_label)):
            if text[count][i] == letters[pred_label[i]]:
                acc += 1
            total += 1
        count += 1

        print(''.join(letters[pred_label]))       
    print('acc:', acc/total * 1.0)