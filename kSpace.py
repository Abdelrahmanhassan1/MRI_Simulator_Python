import numpy as np

image = [[100, 0, 20, 30], [100, 0, 20, 30],
         [100, 0, 20, 30], [100, 0, 20, 30]]

# vector 3x1
Mo = [[0], [0], [100]]

# 90 degree rotation matrix
R = [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]

# 36 degree rotation matrix
R2 = [[np.cos(np.pi/5), np.sin(np.pi/5), 0],
      [-np.sin(np.pi/5), np.cos(np.pi/5), 0],
      [0, 0, 1]]

# 40 degree rotation matrix
R3 = [[np.cos(np.pi/9), np.sin(np.pi/9), 0],
      [-np.sin(np.pi/9), np.cos(np.pi/9), 0],
      [0, 0, 1]]

x1 = np.matmul(R, Mo)
x2 = np.matmul(R2, x1)
x3 = np.matmul(R3, x2)

print(x1)
print(x2)
print(x3)
