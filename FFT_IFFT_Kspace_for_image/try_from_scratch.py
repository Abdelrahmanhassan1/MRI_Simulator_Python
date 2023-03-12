import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math
image = np.array([[100, 30, 50  ], [85, 21 , 30 ],[54 , 78 , 90  ]])
print(f"image = {image}")
ROWS, COLS = image.shape
print(f"ROWS of image = {ROWS} , COLS of image = {COLS}")

####create vector for each pixel
image_to_matrix = np.zeros((ROWS, COLS, 3))
for row in range (ROWS):
    for col in range(COLS):
        image_to_matrix[row,col] = [0, 0, image[row, col]]

print(f"image_to_matrix = {image_to_matrix}")

###create k space
k_space = np.zeros((ROWS, COLS), dtype=complex)
print(f"K space = {k_space}")

####apply RF to make M0 in plane XY
R = np.array([[1, 0, 0],
                  [0, math.cos(math.radians(-90)), -math.sin(math.radians(-90))],
                  [0, math.sin(math.radians(-90)), math.cos(math.radians(-90))]])

for row in range(ROWS):
    for col in range(COLS):
        image_to_matrix[row, col] = np.dot(R, image_to_matrix[row, col])

print(f"image in XY plane = {image_to_matrix}")



Gy_phases = np.arange ( (2*np.pi)/ROWS ,(2*np.pi) + (2*np.pi)/ROWS, (2*np.pi)/ROWS)
Gx_phases = np.arange ( (2*np.pi)/COLS ,(2*np.pi) + (2*np.pi)/COLS , (2*np.pi)/COLS)
##divide phses for x , y
print(f"Gy_phases = {Gy_phases}")
print(f"Gx_phases = {Gx_phases}")
row_Kspace = -1
col_Kspace = -1
##apply Yphases & Xphases
for  phase_y in  Gy_phases:
    row_Kspace = row_Kspace + 1
    matrix_in_xy = image_to_matrix.copy()
    print(f"phase_y = {phase_y}")
    for row ,phase_y_C_divided in zip(range(ROWS ) , np.arange ( phase_y/ROWS ,phase_y +phase_y/ROWS, phase_y/ROWS) ) :
        Ry = np.array([[math.cos(math.radians(phase_y_C_divided)), -math.sin(math.radians(phase_y_C_divided)), 0],
                       [math.sin(math.radians(phase_y_C_divided)), math.cos(math.radians(phase_y_C_divided)), 0],
                       [0, 0, 1]])
        print(f"Ry = {Ry}")
        print(f"phase_y_C_divided = {phase_y_C_divided}")
        for col in range(COLS):
            print(f"row = {row} , col = {col}")
            matrix_in_xy[row,col] = np.dot(Ry, matrix_in_xy[row, col])
    col_for_x = -1
    for phase_x in Gx_phases:
        print(f"phase_x = {phase_x}")
        Rx = np.array([[math.cos(math.radians(phase_x)), -math.sin(math.radians(phase_x)), 0],
                       [math.sin(math.radians(phase_x)), math.cos(math.radians(phase_x)), 0],
                       [0, 0, 1]])
        col_for_x = col_for_x + 1
        for row in range(ROWS):
            print(f" row = {row} , col_for_x = {col_for_x}")
            matrix_in_xy[row, col_for_x] = np.dot(Rx, matrix_in_xy[row, col_for_x])
        col_Kspace = col_Kspace + 1
        if (col_Kspace == 3):
            col_Kspace = 0

        for i in range(ROWS):
            for j in range(COLS):

                print(f"row_Kspace = {row_Kspace} , col_Kspace = {col_Kspace}")
                k_space[row_Kspace, col_Kspace] += np.sqrt(matrix_in_xy[i,j,0] ** 2 + matrix_in_xy[i,j,1] ** 2) * np.exp(complex(0, -(phase_x * j + phase_y * i)))
        print(f"k_space = {k_space}")


img_c3 = np.fft.ifftshift(k_space)
img_c4 = np.fft.ifft2(img_c3)
print(f"img_c4 = {img_c4}")
for row in range(ROWS):
    for col in range (COLS):
        img_c4[row, col] = np.sqrt(img_c4[row, col].real ** 2 + img_c4[row, col].imag ** 2)
print(f"img_c4 = {img_c4}")
# print(f"matrix_in_xy[1, 0] = {matrix_in_xy[1, 0]}")
# print(f"matrix_in_xy[i, j, 0] = {matrix_in_xy[1, 0, 0]} , matrix_in_xy[i, j, 1] = {matrix_in_xy[1, 0, 1]}")


# print(f"RF to make M0 in plane XY = {R} ")
# print(f"np.cos(80) = {np.cos(80)}")
# print(f"np.cos(np.radians(80)) = {np.cos(np.radians(80))}")
# print(f"np.cos(np.pi*2/9) = {np.cos((np.pi*2)/9)}")
# print(f"np.cos(np.radians(np.pi/2)) = {np.cos(np.radians((3.141592653589793*2)/9))}")
# print(f"Pi = {np.pi}")
plt.subplot(121), plt.imshow(image, "gray"), plt.title("image")
plt.subplot(122), plt.imshow(abs(img_c4), "gray"), plt.title("reconstructed image")
#
plt.show()