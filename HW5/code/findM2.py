# ##################################################################### #
# 16720B: Computer Vision Homework 5
# Carnegie Mellon University
# Oct. 26, 2020
# ##################################################################### #

import numpy as np
import helper
import submission as sub

'''
Q3.3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_5.npz
'''


def test_M2_solution(pts1, pts2, intrinsics):
	'''
	Estimate all possible M2 and return the correct M2 and 3D points P
	:param pred_pts1:
	:param pred_pts2:
	:param intrinsics:
	:return: M2, the extrinsics of camera 2
			 C2, the 3x4 camera matrix
			 P, 3D points after triangulation (Nx3)
	'''
	K1 = intrinsics['K1']
	K2 = intrinsics['K2']

	M1 = np.hstack(( np.identity(3), np.zeros((3,1)) ))

	M = 640
	F = sub.eightpoint(pts1, pts2, M)
	E = sub.essentialMatrix(F, K1, K2)
	M2 = helper.camera2(E)

	C1 = K1 @ M1
	err_min = float('inf')
	for i in range(M2.shape[-1]):
		temp_C2 = K2 @ M2[:,:,i]
		temp_P, err = sub.triangulate(C1, pts1, temp_C2, pts2)

		if err < err_min:
		# if np.min(temp_P[:,-1]) > 0:
			C2 = temp_C2
			P = temp_P
			M2_single = M2[:,:,i]
		# print('errors:', err)
	return M2_single, C2, P


if __name__ == '__main__':
	data = np.load('../data/some_corresp.npz')
	pts1 = data['pts1']
	pts2 = data['pts2']
	intrinsics = np.load('../data/intrinsics.npz')

	M2, C2, P = test_M2_solution(pts1, pts2, intrinsics)
	np.savez('q3_5', M2=M2, C2=C2, P=P)
