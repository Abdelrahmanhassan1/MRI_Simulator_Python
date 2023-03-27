import numpy as np


# make 4x4 matrix
matrix = [[1, 2, 3, 4],
          [5, 6, 7, 8],
          [9, 10, 11, 12],
          [13, 14, 15, 16]]

top = 0
bottom = len(matrix) - 1
left = 0
right = len(matrix[0]) - 1
direction = "right"

indicesVisisted = np.array([])

while top <= bottom and left <= right:
    if direction == "right":
        for i in range(left, right+1):

            indicesVisisted = np.append(indicesVisisted, [top, i])
        top += 1
        direction = "down"
    elif direction == "down":
        for i in range(top, bottom+1):

            indicesVisisted = np.append(indicesVisisted, [i, right])
        right -= 1
        direction = "left"
    elif direction == "left":
        for i in range(right, left-1, -1):

            indicesVisisted = np.append(indicesVisisted, [bottom, i])
        bottom -= 1
        direction = "up"
    elif direction == "up":
        for i in range(bottom, top-1, -1):

            indicesVisisted = np.append(indicesVisisted, [i, left])
        left += 1
        direction = "right"

indicesVisisted = indicesVisisted.reshape(-1, 2)
# reverse the indices
indicesVisisted = indicesVisisted[::-1]
print(indicesVisisted)
