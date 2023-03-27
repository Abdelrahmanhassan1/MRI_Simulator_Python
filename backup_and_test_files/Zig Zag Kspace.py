import numpy as np


matrix = [[1, 2, 3, 4, 5],
          [6, 7, 8, 9, 10],
          [11, 12, 13, 14, 15],
          [16, 17, 18, 19, 20],
          [21, 22, 23, 24, 25]]

n = len(matrix)
going_down = True
indicesVisited = np.array([])

for i in range(n):
    if going_down:
        for j in range(n):
            indicesVisited = np.append(indicesVisited, [i, j])
        going_down = False
    else:
        for j in range(n-1, -1, -1):
            indicesVisited = np.append(indicesVisited, [i, j])
        going_down = True


indicesVisited = indicesVisited.reshape(-1, 2)
print(indicesVisited)
