# # make 4x4 matrix
# import numpy as np


import numpy as np


matrix = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12],
                   [13, 14, 15, 16]])

n = len(matrix)
row = col = n // 2

# main arrows from the center
# top Right
topRight = np.array([])
x = row
y = col
while x > 0 and y < n - 1:
    x -= 1
    y += 1
    topRight = np.append(topRight, [x, y])

# bottom Right
bottomRight = np.array([])
x = row
y = col
while x < n - 1 and y < n - 1:
    x += 1
    y += 1
    bottomRight = np.append(bottomRight, [x, y])

# bottom Left
bottomLeft = np.array([])
x = row
y = col
while x < n - 1 and y > 0:
    x += 1
    y -= 1
    bottomLeft = np.append(bottomLeft, [x, y])

# top Left
topLeft = np.array([])
x = row
y = col
while x > 0 and y > 0:
    x -= 1
    y -= 1
    topLeft = np.append(topLeft, [x, y])

# top
top = np.array([])
x = row
y = col
while x > 0:
    x -= 1
    top = np.append(top, [x, y])

# right
right = np.array([])
x = row
y = col
while y < n - 1:
    y += 1
    right = np.append(right, [x, y])

# bottom
bottom = np.array([])
x = row
y = col
while x < n - 1:
    x += 1
    bottom = np.append(bottom, [x, y])

# left
left = np.array([])
x = row
y = col
while y > 0:
    y -= 1
    left = np.append(left, [x, y])

# combine all the arrays
allArrays = np.concatenate(
    (topRight, bottomRight, bottomLeft, topLeft, top, right, bottom, left))

# add the center to the first index of the array
allArrays = np.insert(allArrays, 0, [row, col])
# reshape the array
allArrays = allArrays.reshape(-1, 2)


# import threading

# import numpy as np

# matrix = np.array([[1, 2, 3, 4],
#                    [5, 6, 7, 8],
#                    [9, 10, 11, 12],
#                    [13, 14, 15, 16]])

# n = len(matrix)
# row = col = n // 2


# def top_right():
#     global topRight
#     topRight = np.array([])
#     x = row
#     y = col
#     while x > 0 and y < n - 1:
#         x -= 1
#         y += 1
#         topRight = np.append(topRight, [x, y])


# def bottom_right():
#     global bottomRight
#     bottomRight = np.array([])
#     x = row
#     y = col
#     while x < n - 1 and y < n - 1:
#         x += 1
#         y += 1
#         bottomRight = np.append(bottomRight, [x, y])


# def bottom_left():
#     global bottomLeft
#     bottomLeft = np.array([])
#     x = row
#     y = col
#     while x < n - 1 and y > 0:
#         x += 1
#         y -= 1
#         bottomLeft = np.append(bottomLeft, [x, y])


# def top_left():
#     global topLeft
#     topLeft = np.array([])
#     x = row
#     y = col
#     while x > 0 and y > 0:
#         x -= 1
#         y -= 1
#         topLeft = np.append(topLeft, [x, y])


# def top():
#     global top
#     top = np.array([])
#     x = row
#     y = col
#     while x > 0:
#         x -= 1
#         top = np.append(top, [x, y])


# def right():
#     global right
#     right = np.array([])
#     x = row
#     y = col
#     while y < n - 1:
#         y += 1
#         right = np.append(right, [x, y])


# def bottom():
#     global bottom
#     bottom = np.array([])
#     x = row
#     y = col
#     while x < n - 1:
#         x += 1
#         bottom = np.append(bottom, [x, y])


# def left():
#     global left
#     left = np.array([])
#     x = row
#     y = col
#     while y > 0:
#         y -= 1
#         left = np.append(left, [x, y])


# # create threads for each function
# threads = []
# for func in [top_right, bottom_right, bottom_left, top_left, top, right, bottom, left]:
#     t = threading.Thread(target=func)
#     t.start()
#     threads.append(t)

# # wait for all threads to finish
# for t in threads:
#     t.join()


# # combine all the arrays
# allArrays = np.concatenate(
#     (topRight, bottomRight, bottomLeft, topLeft, top, right, bottom, left))

# # add the center to the first index of the array
# allArrays = np.insert(allArrays, 0, [row, col])

# # reshape the array
# allArrays = allArrays.reshape(-1, 2)
