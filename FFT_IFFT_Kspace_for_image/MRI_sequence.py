import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math
# image = np.array([[1,2,3],[4,5,6],[7,8,9]])
image = cv2.imread('test.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print(f"image = {image}")
ROWS, COLS = image.shape
print(f"ROWS of image = {ROWS} , COLS of image = {COLS}")
# create vector for each pixel
image_to_matrix = np.zeros((ROWS, COLS, 3))
for row in range(ROWS):
    for col in range(COLS):
        image_to_matrix[row, col] = [0, 0, image[row, col]]
print(f"image_to_matrix = {image_to_matrix}")

# create k space
k_space = np.zeros((ROWS, COLS), dtype=complex)
print(f"K space = {k_space}")

# apply RF to make M0 in plane XY
R = np.array([[1, 0, 0],
              [0, math.cos(math.radians(-90)), -math.sin(math.radians(-90))],
              [0, math.sin(math.radians(-90)), math.cos(math.radians(-90))]])

for row in range(ROWS):
    for col in range(COLS):
        image_to_matrix[row, col] = np.dot(R, image_to_matrix[row, col])

print(f"image in XY plane = {image_to_matrix}")

X_Gradient_step = 360 / COLS
Y_Gradient_step = 360 / ROWS
print(
    f"X_Gradient_step = {X_Gradient_step} , Y_Gradient_step = {Y_Gradient_step}")
# Apply Gradient for Y
for Y_rows_total_gradient in range(ROWS):

    # take copy as every time use Y gradient apply every thing for to matrix after move to x-y plane
    matrix_after_Y_G = image_to_matrix.copy()
    print(
        f"gradient value before multi to each Row = { Y_rows_total_gradient * Y_Gradient_step}")
    for row in range(ROWS):
        angle = Y_rows_total_gradient * Y_Gradient_step * row
        print(
            f"gradient value iterate = {row}  after multi to each Row = {angle}")
        Ry = np.array([[math.cos(math.radians(angle)), -math.sin(math.radians(angle)), 0],
                       [math.sin(math.radians(angle)), math.cos(
                           math.radians(angle)), 0],
                       [0, 0, 1]])
        for col in range(COLS):
            print(f"row = {row} , col = {col}")
            matrix_after_Y_G[row, col] = np.dot(Ry, matrix_after_Y_G[row, col])
            print(f"matrix_in_xy[row,col] = {matrix_after_Y_G[row, col]}")

    # Apply Gradient for X
    for X_cols_total_gradient in range(COLS):
        matrix_after_Y_G_X_G = matrix_after_Y_G.copy()
        print(
            f"gradient value before multi to each Column = {X_cols_total_gradient * X_Gradient_step}")
        for col in range(COLS):
            angle = X_cols_total_gradient * X_Gradient_step * col
            print(
                f"gradient value iterate = {col}  after multi to each Row = {angle}")
            Rx = np.array([[math.cos(math.radians(angle)), -math.sin(math.radians(angle)), 0],
                           [math.sin(math.radians(angle)), math.cos(
                               math.radians(angle)), 0],
                           [0, 0, 1]])
            for row in range(ROWS):
                print(f"row = {row} , col = {col}")
                matrix_after_Y_G_X_G[row, col] = np.dot(
                    Rx, matrix_after_Y_G_X_G[row, col])
                print(
                    f"matrix_in_xy[row,col] = {matrix_after_Y_G_X_G[row, col]}")

        x_sum = 0
        y_sum = 0
        for i in range(ROWS):
            for j in range(COLS):
                x_sum += matrix_after_Y_G_X_G[i, j, 0]
                y_sum += matrix_after_Y_G_X_G[i, j, 1]
        k_space[Y_rows_total_gradient, X_cols_total_gradient] = complex(
            y_sum, x_sum)  # complex k_space np.sqrt(x_sum**2 + y_sum**2)
        print(f" matrix_in_xy after k space =  {matrix_after_Y_G_X_G}")

        print(f"k_space = {k_space}")

img_c3 = np.fft.ifftshift(k_space)
img_c5 = np.fft.fftshift(k_space)
img_c4 = np.fft.ifft2(k_space)


plt.subplot(151), plt.imshow(image, "gray"), plt.title("image")
plt.subplot(152), plt.imshow(
    np.log(1+np.abs(k_space)), "gray"), plt.title("K space ")
plt.subplot(153), plt.imshow(
    np.log(1+np.abs(abs(img_c5))), "gray"), plt.title(" shift")
plt.subplot(154), plt.imshow(np.log(1+np.abs(abs(img_c3))),
                             "gray"), plt.title("inverse shift")
plt.subplot(155), plt.imshow(
    abs(img_c4), "gray"), plt.title("reconstructed image")
#
plt.show()
