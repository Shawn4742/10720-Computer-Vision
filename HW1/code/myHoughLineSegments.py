import glob
import os.path as osp
import numpy as np
import cv2

# Refs of Draw lines, see
# https://github.com/nicwigs/491/blob/af05c71d647bf199224cf9d3a58dc06af04be6be/proj1/python/drawLine.py
def drawLines(img, lines):

	if img.ndim == 2:
		img = np.expand_dims(img, axis=2).repeat(3, axis=2)

	if img.dtype != np.float64:
		img = np.float64(img) / 255

	h, w, _ = img.shape
	for line in lines:
		start = line[0]
		end = line[1]

		delta = 0.05/ np.linalg.norm(np.array(end) - np.array(start))
		t = np.arange(0, 1, delta)

		x = start[0] + t * (end[0] - start[0])
		y = start[1] + t * (end[1] - start[1])

		x = np.round(x)
		y = np.round(y)

		# 有必要？
		# x = np.clip(x, 0, w-1)
		# y = np.clip(y, 0, h-1)

		r = img[:,:,0]
		g = img[:,:,1]
		b = img[:,:,2]

		r[np.int64(y), np.int64(x)] = 0.0
		g[np.int64(y), np.int64(x)] = 1.0
		b[np.int64(y), np.int64(x)] = 0.0

		r = np.expand_dims(r, axis=2)
		g = np.expand_dims(g, axis=2)
		b = np.expand_dims(b, axis=2)

		img = np.concatenate( (r,g,b), axis=2 )
	return img

def houghpixel():
	pass

def isValidRange(x, y, w, h):
	if 0 <= x < w and 0 <= y < h:
		return True
	return False

def isValidPixel(x, y, img, w, h):
	if isValidRange(x, y, w, h) and img[y][x] > 0: 
		return True
	return False

def getLength(start, end):
	return np.sqrt( np.square(start[0] - start[1]) + np.square(end[0] - end[1]))

def myHoughLineSegments(img_in, edgeimage, peakRho, peakTheta, rhosscale, thetasscale):
	# Your implemention

	# draw horizontally and vertically

	minLength = 20
	h, w, _ = img_in.shape
	# print(img_in.shape): (480, 640, 3)
	lines = []

	# print(peakRho)
	# print(peakTheta)

	for k in range(len(peakRho)):
		rho = rhosscale[peakRho[k]]
		theta = thetasscale[peakTheta[k]]

		currStart = [0, 0]
		currEnd = [0, 0]
		isDrawing = False

		# horizontally or vertically 
		# loop for each vertical line x = x
		for x in range(w):

			# compute y value:
			# by rho = x * np.cos(theta) - y * np.sin(theta)
			y = -rho/np.sin(theta) + x/np.tan(theta)
			y = np.round(y)

			#  for valid index purpose
			x = np.int64(x)
			y = np.int64(y)

			isValid_xy = isValidPixel(x, y, edgeimage, w, h)
			
			if isDrawing and isValid_xy:
				currEnd = [x, y]
				continue

			if not isDrawing and not isValid_xy: continue

			# start drawing
			if not isDrawing and isValid_xy:
				currStart = [x, y]
				isDrawing = True
				continue

			# end drawing
			if isDrawing and not isValid_xy:
				if isValidRange(x, y, w, h): currEnd = [x, y]
				if getLength(currStart, currEnd) > minLength: lines.append([ currStart, currEnd ])
				isDrawing = False

	img_output = drawLines(img_in, lines)
	return np.uint8(255.0 * img_output)