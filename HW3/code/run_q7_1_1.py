import numpy as np
import scipy.io
# from nn import *
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader

train_data = scipy.io.loadmat('../data/nist36_train.mat')
test_data = scipy.io.loadmat('../data/nist36_test.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
test_x, test_y = test_data['test_data'], test_data['test_labels']

N, D = train_x.shape
_, num_classes = train_y.shape


train_x = torch.from_numpy(train_x).type(torch.float)
train_y = torch.from_numpy(train_y)
test_x = torch.from_numpy(test_x).type(torch.float)
test_y = torch.from_numpy(test_y)

max_iters = 51
# pick a batch size, learning rate
batch_size = 16
learning_rate = 1e-3
hidden_size = 64

batches = DataLoader(TensorDataset(train_x, train_y), shuffle=True, batch_size=batch_size)

model = torch.nn.Sequential(
    torch.nn.Linear(D, hidden_size),
    torch.nn.Sigmoid(),
    torch.nn.Linear(hidden_size, num_classes),
    # torch.nn.Softmax()
)
# print(model)

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
    for xb, yb in batches:
        # training loop can be exactly the same as q2!
        ##########################
        ##### your code here #####
        ##########################

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
    with torch.no_grad():
        output = model(test_x)

        # acc 
        pred_y = torch.argmax(output, axis=1)
        true_y = torch.argmax(test_y, axis=1)
        acc = torch.sum(pred_y == true_y).item() / test_x.size()[0]

        # loss
        loss = torch.nn.CrossEntropyLoss()(output, true_y).item()

    test_loss.append(loss)
    test_acc.append(acc)
    if itr % 5 == 0:
        print("itr: {:02d} \t test_loss: {:.2f} \t test_acc : {:.2f}".format(itr, loss, acc))



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
