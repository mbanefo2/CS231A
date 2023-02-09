import numpy as np

M = np.array([[1,2,3], [4,5,6], [7,8,9], [0,2,2]])
a = np.array([[1], [1], [0]])

tile_a = np.tile(a.T, (4,1))
print(a.T.shape)
print(tile_a.shape)
