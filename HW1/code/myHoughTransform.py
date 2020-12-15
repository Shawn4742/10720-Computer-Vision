import cv2
import numpy as np

# Ref:
# https://teaching.csse.uwa.edu.au/courses/CITS4240/Labs/Lab6/lab6.html

def myHoughTransform(InputImage, rho_resolution, theta_resolution):
	# Your implemention

	rows, colns = InputImage.shape
	max_rho = np.sqrt( rows * rows + colns * colns )

	# 0 <= theta <= pi
	# or
	# -pi/2 <= theta <= pi/2 ?
	thetas = np.arange(0, np.pi, theta_resolution/180 * np.pi )

	# -D <= rho <= D
	rhos = np.arange(-max_rho, max_rho, rho_resolution)

	H = np.zeros( (len(rhos), len(thetas)) )
	# print(H.shape)

	idx = np.where(InputImage > 0)
	# print(idx)
	for i in range(len(idx[0])):
		for j in range(len(thetas)):
			# i -> y, j -> x
			y = idx[0][i]
			x = idx[1][i]

			# The origin is top left? so y -> -y
			# rho = np.round(x * np.cos(theta) - y * np.sin(theta))
			rho = x * np.cos(thetas[j]) - y * np.sin(thetas[j])

			# print(x, y)
			# print(theta)
			# print(rho)
			# print('-----')

			# H_x = np.where(rho == rhos)[0][0]
			H_x = (np.abs(rho - rhos)).argmin()
			# H_y = np.where(theta == thetas)[0][0]
			H[H_x][j] += 1

	return H, rhos, thetas