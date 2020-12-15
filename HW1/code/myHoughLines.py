import numpy as np



def myHoughLines(H, nLines):
    # Your implemention

	# padding first, to avoid "out of index"
	H_pad = np.pad(H, ((1, 1), (1, 1)), mode='constant', constant_values = 0)
	rhos = []
	thetas = []

	for n in range(nLines):
		idx = np.where(H_pad == H_pad.max())
		# print(idx)
		# only pock the first idx
		x = idx[0][0]
		y = idx[1][0]

		# make neighbors as 0
		H_pad[x-1:x+1, y-1:y+1] = 0

		rhos.append(x-1)
		thetas.append(y-1)

	return rhos,thetas