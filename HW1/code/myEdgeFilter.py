import cv2
import numpy as np
import numpy.matlib as npm
import math

from GaussianKernel import Gauss2D
from myImageFilterX import myImageFilterX
from myImageFilter import myImageFilter

def myEdgeFilter(img0, sigma):
	# Your implemention
	hsize = 2*np.ceil(3*sigma) + 1
	f_guassian = Gauss2D( (hsize, hsize), sigma )
	f_sober_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]] )
	f_sober_x = f_sober_y.T

	img0 = myImageFilterX(img0, f_guassian)
	Ix =  myImageFilterX(img0, f_sober_x)
	Iy = myImageFilterX(img0, f_sober_y)

	#  compute magnitudes
	Io = np.sqrt( np.square(Ix) + np.square(Iy) )
	# remove edge values
	# Io[0, :] = 0
	# Io[:, 0] = 0
	# Io[-1, :] = 0
	# Io[:, -1] = 0

	Img1 = Io
	for i in range(1, Img1.shape[0]-1):
		for j in range(1, Img1.shape[1]-1):
			if Ix[i][j] == 0:
				angle = np.pi/2
			else:
				angle = np.arctan(Iy[i][j]/Ix[i][j])

			# arctan return in [-pi/2, pi/2]
			# 0 degree
			if np.pi/8 >= np.abs(angle):				
				if Io[i][j+1] > Io[i][j] or Io[i][j-1] > Io[i][j]:
					Img1[i][j] = 0

			# 45 degree
			elif np.pi/8 <= angle < np.pi/8*3:					
				if Io[i+1][j+1] > Io[i][j] or Io[i-1][j-1] > Io[i][j]:
					Img1[i][j] = 0

			# 90 degree
			elif np.abs(angle) >= np.pi/8*3:					
				if Io[i-1][j] > Io[i][j] or Io[i+1][j] > Io[i][j]:
					Img1[i][j] = 0

			# 135 degree
			elif -np.pi/8*3 < angle <= - np.pi/8:					
				if Io[i-1][j+1] > Io[i][j] or Io[i+1][j-1] > Io[i][j]:
					Img1[i][j] = 0

			else:
				print('No degree matched!')

	# remove edge values
	# Maybe useless, the problem is related to np.pad on the edge values
	Img1[0, :] = 0
	Img1[:, 0] = 0
	Img1[-1, :] = 0
	Img1[:, -1] = 0
	
	return Img1,Io,Ix,Iy


