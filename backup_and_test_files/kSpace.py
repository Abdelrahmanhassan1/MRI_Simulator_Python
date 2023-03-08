# # import numpy as np

# # image = [[100, 0, 20, 30], [100, 0, 20, 30],
# #          [100, 0, 20, 30], [100, 0, 20, 30]]

# # # vector 3x1
# # Mo = [[0], [0], [100]]

# # # 90 degree rotation matrix
# # R = [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]

# # # 36 degree rotation matrix
# # R2 = [[np.cos(np.pi/5), np.sin(np.pi/5), 0],
# #       [-np.sin(np.pi/5), np.cos(np.pi/5), 0],
# #       [0, 0, 1]]

# # # 40 degree rotation matrix
# # R3 = [[np.cos(np.pi/9), np.sin(np.pi/9), 0],
# #       [-np.sin(np.pi/9), np.cos(np.pi/9), 0],
# #       [0, 0, 1]]

# # x1 = np.matmul(R, Mo)
# # x2 = np.matmul(R2, x1)
# # x3 = np.matmul(R3, x2)

# # print(x1)
# # print(x2)
# # print(x3)

import numpy as np
import matplotlib.pyplot as plt
# image = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# rows, columns = image.shape
# print(rows, columns)
# # Define the vector you want to flip as a NumPy array
# Mo = []
# k_space = []

# for phase in range(0, 360-36, 36):
#     sum = 0
#     for i in range(rows):
#         for j in range(columns):
#             Mo = [0, 0, image[i, j]]
#             # flipping each pixel in the matrix by angle 90 degrees
#             theta = np.pi/2  # angle in radians
#             R = np.array([[1, 0, 0],
#                           [0, np.cos(theta), -np.sin(theta)],
#                           [0, np.sin(theta), np.cos(theta)]])
#             # Multiply the vector v by the rotation matrix R to get the flipped vector v_flipped
#             Mo_flipped_xy_plane = np.round(np.dot(R, Mo), 2)

#             # applying Gy gradient with phase = 36 degrees
#             theta = phase * np.pi/180  # angle in radians
#             R2 = np.array([[np.cos(theta), np.sin(theta), 0],
#                            [-np.sin(theta), np.cos(theta), 0],
#                            [0, 0, 1]])
#             Mxy = np.round(np.dot(R2, Mo_flipped_xy_plane), 2)
#             # applying Gx gradient with phase = 36 degrees
#             theta = phase * np.pi/180  # angle in radians
#             R3 = np.array([[np.cos(theta), np.sin(theta), 0],
#                            [-np.sin(theta), np.cos(theta), 0],
#                            [0, 0, 1]])
#             Mxy = np.round(np.dot(R3, Mxy), 2)

#             # get the magnitude of the vector
#             M = np.sqrt(Mxy[0]**2 + Mxy[1]**2 + Mxy[2]**2)
#             sum += np.round(M, 2)
#     k_space.append(sum)

# print(np.array(k_space).reshape(rows, columns))

image = np.array([[100, 0], [100, 0]])

rows, columns = image.shape

new_matrix_image = np.zeros((rows, columns, 3))
gx_counter = 0
gy_counter = 0
gx_phases = np.arange(0, 360, 360/rows)
gy_phases = np.arange(0, 360, 360/rows)


def RF_pulse():
    # make a rotation matrix with 90 along x-axis
    theta = np.pi/2  # angle in radians
    R = np.array([[1, 0, 0],
                  [0, np.cos(theta), -np.sin(theta)],
                  [0, np.sin(theta), np.cos(theta)]])
    # loop over each pixel in the image
    for i in range(rows):
        for j in range(columns):
            # define the vector Mo
            Mo = [0, 0, image[i, j]]
            # Multiply the vector v by the rotation matrix R to get the flipped vector v_flipped
            Mo_flipped_xy_plane = np.round(np.dot(R, Mo), 2)
            new_matrix_image[i, j] = Mo_flipped_xy_plane


def Gx_gradient():
    theta = gx_phases[0]

    R = np.array([[1, 0, 0],
                  [0, np.cos(theta), -np.sin(theta)],
                  [0, np.sin(theta), np.cos(theta)]])
    # loop over each pixel in the image
    for i in range(rows):
        for j in range(columns):
            # define the vector Mo
            Mo = [0, 0, image[i, j]]
            # Multiply the vector v by the rotation matrix R to get the flipped vector v_flipped
            Mo_flipped_xy_plane = np.round(np.dot(R, Mo), 2)
            new_matrix_image[i, j] = Mo_flipped_xy_plane


def Gy_gradient():
    theta = gy_phases[0]

    R = np.array([[1, 0, 0],
                  [0, np.cos(theta), -np.sin(theta)],
                  [0, np.sin(theta), np.cos(theta)]])
    # loop over each pixel in the image
    for i in range(rows):
        for j in range(columns):
            # define the vector Mo
            Mo = [0, 0, image[i, j]]
            # Multiply the vector v by the rotation matrix R to get the flipped vector v_flipped
            Mo_flipped_xy_plane = np.round(np.dot(R, Mo), 2)

            new_matrix_image[i, j] = Mo_flipped_xy_plane


points = []


def ReadOut_Signal():
    points = []
    # loop over the image and get the magnitude of the vector
    for i in range(rows):
        for j in range(columns):
            # define the vector Mo
            M_Vector = image[i, j]
            # get the x-value and y-value as a point and store them  in a list and plot them as a points
            point = [M_Vector[0], M_Vector[1]]
            # store the points in a list
            points.append(point)
    # plot the points
    plt.scatter(*zip(*points))
    plt.show()


def sum_vectors_of_new_matrix():
    sum = 0
    for i in range(rows):
        for j in range(columns):
            sum += np.round(new_matrix_image[i, j], 2)
    print(sum)


RF_pulse()
print(new_matrix_image)
Gx_gradient()
print(new_matrix_image)
sum_vectors_of_new_matrix()
