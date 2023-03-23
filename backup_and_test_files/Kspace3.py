# # import numpy as np


import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
# construct 4x4 image
image = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12],
                  [13, 14, 15, 16]])

# image = cv2.imread('../images/shepp_logan_phantom/128px-Shepp_logan.png')
# print(f"image shape: {image.shape}")
# convert the image to numpy array
# image = np.array(image)

plt.imshow(image, cmap='gray')
plt.show()
a_fft = np.fft.fft2(image)

# print the result
print(f"fft2 result:\n {a_fft}")

a_ifft = np.fft.ifft2(a_fft)

# parse the type to be int not complex numbers
a_ifft = a_ifft.real

# plot the image after fft and ifft
# plt.imshow(a_ifft, cmap='gray')
# plt.show()

# print the result
# print(f"ifft2 result: {a_ifft}")

print("=============================================")


def apply_rf_pulse(image, flip_angle):
    rows, columns = image.shape
    # make a rotation matrix with 90 along x-axis
    theta = flip_angle * np.pi/180  # angle in radians

    # rotation along y axis
    R = np.array([[np.cos(theta), 0, np.sin(theta)],
                  [0, 1, 0],
                  [-np.sin(theta), 0, np.cos(theta)]])

    new_3D_matrix_image = np.zeros((rows, columns, 3))
    # loop over each pixel in the image
    for i in range(rows):
        for j in range(columns):
            # define the vector Mo
            Mo = [0, 0, image[i, j]]
            # Multiply the vector v by the rotation matrix R to get the flipped vector v_flipped
            Mo_flipped_xy_plane = np.round(np.dot(R, Mo), 2)
            new_3D_matrix_image[i, j] = Mo_flipped_xy_plane

    return new_3D_matrix_image


# print(apply_rf_pulse(image, 60))

def update_kspace(kspace):
    # plot k_space as image
    plt.rcParams["figure.figsize"] = [7.00, 3.50]

    plt.rcParams["figure.autolayout"] = True

    # make the background white
    plt.rcParams['axes.facecolor'] = 'white'

    # Plot the data using imshow with gray colormap but the background is white
    plt.imshow(kspace, cmap='gray')

    # Display the plot

    plt.show()


def update_image(kspace_2d):
    # kspace_2d_e = np.fft.ifft2(kspace_2d)
    # # print(f"kspace_2d", kspace_2d)
    # kspace_2d_a = np.round(kspace_2d_e.real, 2)
    # # print(f"inverse kspace_2d", kspace_2d)

    # plt.imshow(kspace_2d_a, cmap='gray')
    # plt.show()
    # reverse the 2D fourier transform
    img = np.fft.ifft2(kspace_2d)
    img = np.real(img).astype(np.uint8)
    plt.imshow(img, cmap='gray')
    plt.show()


def apply_sequence(image_after_rf_pulse):
    backup_image = image_after_rf_pulse.copy()
    rows, columns, _ = image_after_rf_pulse.shape
    k_space_2d = np.zeros((rows, columns), dtype=complex)
    k_space = np.ones((rows, columns))
    phases = np.zeros((rows, columns))
    gx_phases = np.arange(0, 360, 360 / rows)
    gy_phases = np.arange(0, 360, 360 / rows)

    # gy_phases = [30]
    # gx_phases = [0, 90, 120]

    for row_index, gy_phase in enumerate(gy_phases):

        for column_index, gx_phase in enumerate(gx_phases):
            image_after_rf_pulse = backup_image.copy()
            for i in range(rows):
                for j in range(columns):
                    pixel_value = image_after_rf_pulse[i, j, 0]
                    # print(
                    #     f"at row index {gy_phase} and column index {gx_phase} pixel value is {pixel_value} ")
                    phase_from_gy = gy_phase * i
                    phase_from_gx = gx_phase * j
                    applied_phase = (phase_from_gx + phase_from_gy)
                    phases[i, j] = applied_phase
                    applied_phase *= np.pi/180
                    # print(f"applied_phase", applied_phase)
                    new_x_value = pixel_value * np.cos(applied_phase)
                    new_y_value = pixel_value * np.sin(applied_phase)
                    image_after_rf_pulse[i, j, 0] = new_x_value
                    image_after_rf_pulse[i, j, 1] = new_y_value
            print(f"image_after_rf_pulse", image_after_rf_pulse)
            # print(f"phases \n", phases)
            # print(f"image_after_rf_pulse", image_after_rf_pulse)
            # sum the image vectors
            sum = np.round(np.sum(image_after_rf_pulse, axis=(0, 1)), 2)
            # print(f"sum", sum)
            # print(sum)

            # get the magnitude of the vector
            # M = np.sqrt(sum[0]**2 + sum[1]**2)
            # print(M)
            # k_space[row_index, column_index] = M
            k_space_2d[row_index][column_index] = np.round(
                sum[0], 2) - 1j * np.round(sum[1], 2)
            # magnitude of the vector
            k_space[row_index, column_index] = np.sqrt(sum[0]**2 + sum[1]**2)
            # print(f"k_space \n", k_space)
            # update_kspace(k_space)
    # update_image(k_space_2d)
    img = np.fft.ifft2(k_space_2d)
    img = np.real(img).astype(np.uint8)
    plt.imshow(img, cmap='gray')
    plt.show()

    # print("=============================================")
    print(f"k_space 2d \n", k_space_2d)
    print(f"k_space \n", k_space)

    # plt.rcParams["figure.figsize"] = [7.00, 3.50]

    # plt.rcParams["figure.autolayout"] = True

    # Plot the data using imshow with gray colormap
    # plt.imshow(k_space, cmap='gray')

    # # Display the plot
    # plt.show()

    # calculate inverse fourir to 2d k_space

    # k_space_2d = np.fft.ifft2(k_space_2d)
    # k_space_2d = k_space_2d.real
    # print(f"inverse k_space_2d", k_space_2d)
    # k_space = np.fft.fftshift(k_space)
    # plt.imshow(k_space_2d, cmap='gray')
    # plt.show()


apply_sequence(apply_rf_pulse(image, 90))

# # make a rotation matrix with 90 along x-axis
# theta = np.pi/2  # angle in radians

# # rotation along y axis
# R = np.array([[np.cos(theta), 0, np.sin(theta)],
#               [0, 1, 0],
#               [-np.sin(theta), 0, np.cos(theta)]])

# new_3D_matrix_image = np.zeros((rows, columns, 2))
# # loop over each pixel in the image
# for i in range(rows):
#     for j in range(columns):
#         # define the vector Mo
#         Mo = [0, 0, image_after_rf_pulse[i, j]]
#         # Multiply the vector v by the rotation matrix R to get the flipped vector v_flipped
#         Mo_flipped_xy_plane = np.round(np.dot(R, Mo), 2)
#         new_3D_matrix_image[i, j] = Mo_flipped_xy_plane[0:2]

# return new_3D_matrix_image

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


# image = np.array([[100, 0], [100, 0]])

# rows, columns = image.shape
# print(f"shape= {image.shape}")

# new_matrix_image = np.zeros((rows, columns, 3))
# gx_counter = 0
# gy_counter = 0
# gx_phases = np.arange(0, 360, 360/rows)
# gy_phases = np.arange(0, 360, 360/rows)


# def RF_pulse():
#     # make a rotation matrix with 90 along x-axis
#     theta = np.pi/2  # angle in radians
#     R = np.array([[1, 0, 0],
#                   [0, np.cos(theta), -np.sin(theta)],
#                   [0, np.sin(theta), np.cos(theta)]])
#     # loop over each pixel in the image
#     for i in range(rows):
#         for j in range(columns):
#             # define the vector Mo
#             Mo = [0, 0, image[i, j]]
#             # Multiply the vector v by the rotation matrix R to get the flipped vector v_flipped
#             Mo_flipped_xy_plane = np.round(np.dot(R, Mo), 2)
#             new_matrix_image[i, j] = Mo_flipped_xy_plane


# def Gx_gradient():
#     theta = gx_phases[0]

#     R = np.array([[1, 0, 0],
#                   [0, np.cos(theta), -np.sin(theta)],
#                   [0, np.sin(theta), np.cos(theta)]])
#     # loop over each pixel in the image
#     for i in range(rows):
#         for j in range(columns):
#             # define the vector Mo
#             Mo = [0, 0, image[i, j]]
#             # Multiply the vector v by the rotation matrix R to get the flipped vector v_flipped
#             Mo_flipped_xy_plane = np.round(np.dot(R, Mo), 2)
#             new_matrix_image[i, j] = Mo_flipped_xy_plane


# def Gy_gradient():
#     theta = gy_phases[0]

#     R = np.array([[1, 0, 0],
#                   [0, np.cos(theta), -np.sin(theta)],
#                   [0, np.sin(theta), np.cos(theta)]])
#     # loop over each pixel in the image
#     for i in range(rows):
#         for j in range(columns):
#             # define the vector Mo
#             Mo = [0, 0, image[i, j]]
#             # Multiply the vector v by the rotation matrix R to get the flipped vector v_flipped
#             Mo_flipped_xy_plane = np.round(np.dot(R, Mo), 2)

#             new_matrix_image[i, j] = Mo_flipped_xy_plane


# points = []


# def ReadOut_Signal():
#     points = []
#     # loop over the image and get the magnitude of the vector
#     for i in range(rows):
#         for j in range(columns):
#             # define the vector Mo
#             M_Vector = image[i, j]
#             # get the x-value and y-value as a point and store them  in a list and plot them as a points
#             point = [M_Vector[0], M_Vector[1]]
#             # store the points in a list
#             points.append(point)
#     # plot the points
#     plt.scatter(*zip(*points))
#     plt.show()


# def sum_vectors_of_new_matrix():
#     sum = 0
#     for i in range(rows):
#         for j in range(columns):
#             sum += np.round(new_matrix_image[i, j], 2)
#     print(sum)


# print(f"iamge = {image}\n")
# print(f"new = {new_matrix_image}\n")

# print(f"index 0 = {new_matrix_image[0,1]}\n\n\n\n\n")


# RF_pulse()
# print(f"{new_matrix_image}\n\n")
# Gx_gradient()
# print(new_matrix_image)
# sum_vectors_of_new_matrix()
