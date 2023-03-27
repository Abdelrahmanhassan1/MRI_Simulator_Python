

import numpy as np

matrix = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12],
                  [13, 14, 15, 16]])

a_fft = np.fft.fft2(matrix)

# print the result
print(f"fft2 result:\n {a_fft}")


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

kspaceIndicesTovist = np.array([[0, 0], [0, 1], [0, 2], [0, 3], [1, 0], [1, 1], [
                               1, 2], [1, 3], [2, 0], [2, 1], [2, 2], [2, 3], [3, 0], [3, 1], [3, 2], [3, 3]])

print(kspaceIndicesTovist.shape)


kspace_length = len(kspaceIndicesTovist)
rows, columns, _ = newMatrix.shape
kspaceAngle = 2 * np.pi / rows
k_space_2d = np.zeros((rows, columns), dtype=complex)
k_space = np.ones((rows, columns))
phases = np.zeros((rows, columns))

for index in range(kspace_length):
    u = kspaceIndicesTovist[index, 0]
    v = kspaceIndicesTovist[index, 1]

    image_after_sequence = newMatrix.copy()
    for i in range(rows):
        for j in range(columns):
            x_angle = i * kspaceAngle * u
            y_angle = j * kspaceAngle * v
            angle = x_angle + y_angle

            rotation_matrix = np.array([[np.cos(angle), np.sin(angle), 0],
                                        [-np.sin(angle), np.cos(angle), 0],
                                        [0, 0, 1]])
            Mo_flipped = newMatrix[i, j]
            # apply rotation matrix to the flipped Mo
            Mo_flipped = np.dot(rotation_matrix, Mo_flipped)
            image_after_sequence[i, j] = Mo_flipped
    # print(f"phase matrix: u {u} v {v} \n {phases}")

    # sum the x values of the image after sequence
    sum_x = np.round(np.sum(image_after_sequence[:, :, 0]), 2)
    # sum the y values of the image after sequence
    sum_y = np.round(np.sum(image_after_sequence[:, :, 1]), 2)
    k_space_2d[u, v] = sum_x + 1j * sum_y

print(k_space_2d.shape)
print(k_space_2d)
