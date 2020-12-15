# ##################################################################### #
# 16720B: Computer Vision Homework 5
# Carnegie Mellon University
# Oct. 26, 2020
# ##################################################################### #
import numpy as np
import matplotlib.pyplot as plt
import submission as sub
import helper
import findM2

from mpl_toolkits.mplot3d import Axes3D
'''
Q3.4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''
im1 = plt.imread('../data/im1.png')
im2 = plt.imread('../data/im2.png')

data = np.load('../data/some_corresp.npz')
pts1 = data['pts1']
pts2 = data['pts2']

intrinsics = np.load('../data/intrinsics.npz')
K1 = intrinsics['K1']
K2 = intrinsics['K2']

points = np.load('../data/templeCoords.npz')
x1 = points['x1']
y1 = points['y1']

M = 640

F = sub.eightpoint(data['pts1'], data['pts2'], M)
E = sub.essentialMatrix(F, K1, K2)

x2 = np.zeros_like(x1)
y2 = np.zeros_like(y1)
for i in range(x1.shape[0]):
    x2[i], y2[i] = sub.epipolarCorrespondence(im1, im2, F, x1[i], y1[i])

# for selected points
points1 = np.hstack((x1, y1))
points2 = np.hstack((x2, y2))

M1 = np.hstack(( np.identity(3), np.zeros((3,1)) ))
C1 = K1 @ M1

M2, C2, P = findM2.test_M2_solution(points1, points2, intrinsics)
np.savez('q3_4_2.npz', M1 = M1, M2 = M2, C1 = C1, C2 = C2, F = F)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(P[:,0], P[:,1], P[:,2], marker='o')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()