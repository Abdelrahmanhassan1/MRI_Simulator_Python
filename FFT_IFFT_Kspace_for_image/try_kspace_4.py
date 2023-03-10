import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt



# Load the image
def load_image_to_gray():
    image = cv2.imread('test.png')
    # Convert it to grayscale
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)





def create_image_with_vector(gray_image):
    for i in range(gray_image.shape[0]):
        for j in range(gray_image.shape[1]):
            Mo = [0, 0, gray_image[i, j]]
            new_matrix_image[i, j] = Mo


def RF_pulse(theta_RF):
    # make a rotation matrix with 90 along x-axis
    R = np.array([[1, 0, 0],
                  [0, np.cos(theta_RF), -np.sin(theta_RF)],
                  [0, np.sin(theta_RF), np.cos(theta_RF)]])
    # loop over each pixel in the image
    for i in range(image_gray.shape[0]):
        for j in range(image_gray.shape[1]):

            # Multiply the vector v by the rotation matrix R to get the flipped vector v_flipped
            Mo_flipped_xy_plane = np.round(np.dot(R, new_matrix_image[i,j]), 5)
            new_matrix_image[i, j] = Mo_flipped_xy_plane



image_gray =load_image_to_gray()
print(f"image_gray ={image_gray}")
rows,cols = image_gray.shape
print(f"image shape = {rows}*{cols}")
k_space = image_gray.copy()
new_matrix_image = np.zeros((image_gray.shape[0], image_gray.shape[1], 3))
create_image_with_vector(image_gray)
print(f"image after convert each pixel to vector ={new_matrix_image}\n\n ")
RF_pulse(np.pi/2) ### here apply RF pulse
print(f"image_to_vector after apply RF ={new_matrix_image}\n\n ")

Gx_phases = np.arange (0,360  , 360/image_gray.shape[1])
Gy_phases = np.arange (0,360 , 360/image_gray.shape[0])
print(f"Gx_phases = {Gx_phases}")
print(f"Gy_phases = {Gy_phases}")

row_Kspace = -1
col_Kspace = -1

#for i, phase in zip(range(rows), range(0, 360-36, 36)):
for  phase_y in  Gy_phases:
    row_Kspace = row_Kspace + 1
    matrix_in_xy = new_matrix_image.copy()
    Ry = np.array([[np.cos(phase_y), -np.sin(phase_y), 0],
              [np.sin(phase_y), np.cos(phase_y), 0],
              [0, 0, 1]])

    for row in range(rows):
        for col in range(cols):
            matrix_in_xy[row, col] = np.round(np.dot(Ry, matrix_in_xy[row, col]), 5)

    col_for_x = -1
    for phase_x in Gx_phases:
        Rx = np.array([[np.cos(phase_x), -np.sin(phase_x), 0],
                       [np.sin(phase_x), np.cos(phase_x), 0],
                       [0, 0, 1]])
        #matrix_in_xy_x = matrix_in_xy.copy()
        col_for_x = col_for_x + 1
        for row in range(rows):
            matrix_in_xy[row, col_for_x] = np.round(np.dot(Rx, matrix_in_xy[row, col_for_x]), 5)
        col_Kspace = col_Kspace + 1
        if(col_Kspace == 103):
            col_Kspace = 0
        sum = 0
        for i in range(rows):
            for j in range(cols):
                sum += np.round(matrix_in_xy[i, j], 2)
        # print(f" sum = {sum}")
        # print(f"Magnitude of sum = {np.sqrt(sum[0] ** 2 + sum[1] ** 2 + sum[2] ** 2)}")
        print(f"row_Kspace= {row_Kspace} ,col_Kspace = {col_Kspace} ,phase_x =  {phase_x} ,phase_y = {phase_y}  , col_for_x = {col_for_x} ")
        k_space[row_Kspace,col_Kspace] = np.sqrt(sum[0] ** 2 + sum[1] ** 2 ) * np.exp(complex(0,( -1 * ( (phase_x * col_for_x) + (phase_y * row_Kspace) ) )))

count = np.count_nonzero(k_space == 51)
print(f"count = {count}")
print(f"k_space = {k_space}")

plt.figure(figsize=(6.4*5, 4.8*5), constrained_layout=False)
img_c4 = np.fft.ifftshift(k_space)
img_c5 = np.fft.ifft2(img_c4)
img_c6 =  np.fft.ifft2(k_space)

plt.subplot(151), plt.imshow(image_gray, "gray"), plt.title("Original Image")
plt.subplot(152), plt.imshow(k_space, "gray"), plt.title("k_space")
plt.subplot(153), plt.imshow(np.log(1+np.abs(img_c4)), "gray"), plt.title("Spectrum")
plt.subplot(154), plt.imshow(np.log(1+np.abs(img_c5)), "gray"), plt.title("Processed Image")
plt.subplot(155), plt.imshow(np.log(1+np.abs(img_c6)), "gray"), plt.title("Processed Image")

plt.show()