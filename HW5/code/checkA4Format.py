"""
Check the dimensions of function arguments
This is *not* a correctness check
"""
import matplotlib.pyplot as plt
import numpy as np
import submission as sub
import helper

data = np.load('../data/some_corresp.npz')
im1 = plt.imread('../data/im1.png')
im2 = plt.imread('../data/im2.png')

N = data['pts1'].shape[0]
M = 640

# # 3.2.1
F8 = sub.eightpoint(data['pts1'], data['pts2'], M)
assert F8.shape == (3, 3), 'eightpoint returns 3x3 matrix'

# print(F8)
# np.savez('q_3_2_1.npz', F=F8, M=M)
# # visualize
# helper.displayEpipolarF(im1, im2, F8)


# # 3.2.2
# idx = np.random.choice(N, 7, replace=False)
# F7 = sub.sevenpoint(data['pts1'][idx, :], data['pts2'][idx, :], M)
# # F7 = sub.sevenpoint(data['pts1'], data['pts2'], M)
# idx = np.random.choice(data['pts1'].shape[0], 7, replace=False)
# # F7 = sub.sevenpoint(data['pts1'][idx, :], data['pts2'][idx, :], M)
# assert (len(F7) == 1) | (len(F7) == 3), 'sevenpoint returns length-1/3 list'

# for f7 in F7:
# 	assert f7.shape == (3, 3), 'seven returns list of 3x3 matrix'

# 	# # visualize
# 	# helper.displayEpipolarF(im1, im2, f7)

# print(F7)
# np.savez('q_3_2_2.npz', F=F7, M=M, pts1=data['pts1'][idx, :], pts2=data['pts2'][idx, :])
# helper.displayEpipolarF(im1, im2, F7[0])


# # 3.3.2
# C1 = np.concatenate([np.random.rand(3, 3), np.ones([3, 1])], axis=1)
# C2 = np.concatenate([np.random.rand(3, 3), np.ones([3, 1])], axis=1)

# P, err = sub.triangulate(C1, data['pts1'], C2, data['pts2'])
# assert P.shape == (N, 3), 'triangulate returns Nx3 matrix P'
# assert np.isscalar(err), 'triangulate returns scalar err'

# 3.4.1
x2, y2 = sub.epipolarCorrespondence(im1, im2, F8, data['pts1'][0, 0], data['pts1'][0, 1])
assert np.isscalar(x2) & np.isscalar(y2), 'epipolarCoorespondence returns x & y coordinates'


np.savez('q_3_4_1.npz', F=F8, M=M)
helper.epipolarMatchGUI(im1, im2, F8)


# # 3.5.1
# data = np.load('../data/some_corresp_noisy.npz')
# F, inliers = sub.ransacF(data['pts1'], data['pts2'], M)
# assert F.shape == (3, 3), 'ransacF returns 3x3 matrix'
# print('noisy inliers:', np.sum(inliers)/inliers.shape[0])

# # 3.5.2
# r = np.ones([3, 1])
# R = sub.rodrigues(r)
# assert R.shape == (3, 3), 'rodrigues returns 3x3 matrix'

# R = np.eye(3)
# r = sub.invRodrigues(R)
# assert (r.shape == (3,)) or (r.shape == (3, 1)), 'invRodrigues returns 3x1 vector'

# 3.5.3
K1 = np.random.rand(3, 3)
K2 = np.random.rand(3, 3)
M1 = np.concatenate([np.random.rand(3, 3), np.ones([3, 1])], axis=1)
M2 = np.concatenate([np.random.rand(3, 3), np.ones([3, 1])], axis=1)
r2 = np.ones(3)
t2 = np.ones(3)
x = np.concatenate([P.reshape([-1]), r2, t2])
residuals = sub.rodriguesResidual(K1, M1, data['pts1'], K2, data['pts1'], x)
# assert residuals.shape == (4 * N, 1), 'rodriguesResidual returns vector of size 4Nx1'
assert residuals.shape == (4 * N, ), 'rodriguesResidual returns vector of size 4Nx1'

M2, P = sub.bundleAdjustment(K1, M1, data['pts1'], K2, M2, data['pts1'], P)
assert M2.shape == (3, 4), 'bundleAdjustment returns 3x4 matrix M'
assert P.shape == (N, 3), 'bundleAdjustment returns Nx3 matrix P'

print('Format check passed.')
