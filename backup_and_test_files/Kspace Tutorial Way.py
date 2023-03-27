

import numpy as np

matrix = [[1, 2, 3, 4, 5],
          [6, 7, 8, 9, 10],
          [11, 12, 13, 14, 15],
          [16, 17, 18, 19, 20],
          [21, 22, 23, 24, 25]]


def apply_rf_pulse(image, flip_angle):
    rows, columns = image.shape
    # make a rotation matrix with 90 along x-axis
    theta = flip_angle * np.pi / 180  # angle in radians

    # rotation along y axis
    R = np.array([[np.cos(theta), 0, np.sin(theta)],
                  [0, 1, 0],
                  [-np.sin(theta), 0, np.cos(theta)]])

    # define the vector Mo for the entire image
    Mo = np.zeros((rows, columns, 3))
    Mo[:, :, 2] = image

    # Multiply the vector v by the rotation matrix R to get the flipped vector v_flipped
    Mo_flipped_xy_plane = np.round(np.matmul(Mo, R.T), 2)

    return Mo_flipped_xy_plane


newMatrix = apply_rf_pulse(np.array(matrix), 90)


kspaceIndicesTovist = np.array([[0., 0.],
                                [0., 1.],
                                [0., 2.],
                                [0., 3.],
                                [0., 4.],
                                [1., 4.],
                                [1., 3.],
                                [1., 2.],
                                [1., 1.],
                                [1., 0.],
                                [2., 0.],
                                [2., 1.],
                                [2., 2.],
                                [2., 3.],
                                [2., 4.],
                                [3., 4.],
                                [3., 3.],
                                [3., 2.],
                                [3., 1.],
                                [3., 0.],
                                [4., 0.],
                                [4., 1.],
                                [4., 2.],
                                [4., 3.],
                                [4., 4.]], dtype=np.int8)


kspace_length = len(kspaceIndicesTovist)
kspaceAngle = 2 * np.pi / kspace_length
for index in range(kspace_length):
    u = kspaceIndicesTovist[index, 0]
    v = kspaceIndicesTovist[index, 1]
    for i in range(newMatrix.shape[0]):
        for j in range(newMatrix.shape[1]):

            x_angle = u * kspaceAngle
            y_angle = v * kspaceAngle
            total_angle = x_angle + y_angle
            Mo_flipped = newMatrix[u, v]
            # rotation matrix in z axis
            rotation_matrix = np.array([[np.cos(total_angle), -np.sin(total_angle), 0],
                                        [np.sin(total_angle), np.cos(
                                            total_angle), 0],
                                        [0, 0, 1]])
            # apply rotation matrix to the flipped Mo
            Mo_flipped = np.matmul(rotation_matrix, Mo_flipped)
