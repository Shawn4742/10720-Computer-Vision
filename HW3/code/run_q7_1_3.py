import numpy as np
import scipy.io
# from nn import *
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
# from torchsummary import summary

train_data = scipy.io.loadmat('../data/nist36_train.mat')
test_data = scipy.io.loadmat('../data/nist36_test.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
test_x, test_y = test_data['test_data'], test_data['test_labels']

# (, 1024) --> (, 32x32)

N, D = train_x.shape
_, num_classes = train_y.shape


# # reshape directly? Yes,
# train_x2 = np.reshape(train_x, (train_x.shape[0], 32, 32))
# idx = 322
# print(sum(train_x[idx] != train_x2[idx].flatten()))

train_x = np.reshape(train_x, (N, 32, 32))
test_x = np.reshape(test_x, (test_x.shape[0], 32, 32))

train_x = np.expand_dims(train_x, axis=1)
test_x = np.expand_dims(test_x, axis=1)

train_x = torch.from_numpy(train_x).type(torch.float)
train_y = torch.from_numpy(train_y)
test_x = torch.from_numpy(test_x).type(torch.float)
test_y = torch.from_numpy(test_y)

max_iters = 51
# pick a batch size, learning rate
batch_size = 16
learning_rate = 1e-3
hidden_size = 64
num_classes = 36
c_size = 6 * 7*7

train_loader = DataLoader(TensorDataset(train_x, train_y), shuffle=True, batch_size=batch_size)
test_loader = [(test_x, test_y)]


# 28 --> 24 --> 12 --> 6 --> 3
# 32 --> 28 --> 14 --> 7
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
        x = F.relu(x)

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
        true_y = torch.argmax(yb, axis=1)
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
            true_y = torch.argmax(test_y, axis=1)
            acc.append( torch.sum(pred_y == true_y).item() / test_x.size()[0] )
            # loss
            loss += torch.nn.CrossEntropyLoss()(output, true_y).item()

    acc = np.mean(acc)
    test_loss.append(loss)
    test_acc.append(acc)
    if itr % 5 == 0:
        print("itr: {:02d} \t test_loss: {:.2f} \t test_acc : {:.2f}".format(itr, loss, acc))


# # save
# PATH = './cifar_net.pth'
# torch.save(net.state_dict(), PATH)
# # load
# model = Net()
# model.load_state_dict(torch.load(PATH))

# Q3.1 Plots
if True:
    plt.subplot(1,2,1)
    plt.plot(train_acc, 'b', label = 'train')
    plt.plot(test_acc, 'r', label = 'test')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(train_loss, 'b', label = 'train')
    plt.plot(test_loss, 'r', label = 'test')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()

    plt.show()