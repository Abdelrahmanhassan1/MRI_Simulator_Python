import numpy as np

image = np.array([[1, 2], [4, 5]])

# create 3d matrix with rows an columns of image
k_space = np.zeros((image.shape[0], image.shape[1], 3))

k_space[0, 0] = np.array([1, 2, 7])
print(k_space)
