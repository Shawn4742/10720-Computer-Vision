import numpy as np
import skimage.transform

a = np.array([[1, 0],
               [0, 1]])
b = np.array([[4, 1],
              [2, 2]])
c = np.array([4,  0,  2,  5,  2])
print(c[:-3])



# print(np.matmul(a, b))
# print(np.dot(a, b))


# z = []
# l = np.array([[240,58,3], 
#                 [38,200,5]])

# print(l)
# l = skimage.transform.resize(l, (2, 3))
# print(l.shape)
# print(l)
# # print(l.reshape(-1))

# z.append(l)
# z.append(l)
# z.append(l)
# # print(np.asarray(z).shape)

# a = np.array([10,2,3])
# print(np.minimum(a, l), )
# zz = []
# zz.append(a)
# zz.append(a)
# print(np.asarray(zz))

# z = np.dstack(z)
# print(z[:,:,1])
# print(np.square(z))
# print(z[[0,1],[1,2], :])

# idx = np.unravel_index([22, 41, 37], (7,6))
# x = np.array([idx[0], idx[1]]).T
# print(np.max(x, axis=0))

# train_data = np.load("../data/train_data.npz")
# print(len(train_data))
# print(train_data['files'][1])