from collections import Counter
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QPixmap, QImage, QColor, QPainter, QPen, QBrush
from PyQt5.QtCore import Qt, QRect
import cv2
import sys
import numpy as np
import pyqtgraph as pg
from pyqtgraph import *
import heapq
import pyqtgraph.exporters
import matplotlib.pyplot as plt
from mainWindow import Ui_MainWindow
import math


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # phantom image
        self.prev_x = 0
        self.prev_y = 0
        self.ui.comboBox.currentIndexChanged.connect(
            self.handle_image_features_combo_box)
        self.show_image_on_label(
            './images/phantom_modified/480px-Shepp_logan.png')
        self.modify_the_image_intensities_distribution()
        self.ui.phantom_image_label.mousePressEvent = self.handle_mouse_press
        matrix = self.apply_rf_pulse_on_image()
        matrix2 = self.apply_phase_encoding_Gy_gradient(matrix)
        self.apply_freqency_encoding_Gx_gradient(matrix2)

        # MRI Sequence

    @QtCore.pyqtSlot()
    def show_image_on_label(self, image_path):
        self.original_phantom_image = cv2.imread(image_path, 0)
        img = QImage(image_path)
        pixmap = QPixmap.fromImage(img)
        self.ui.phantom_image_label.setPixmap(pixmap)

    def handle_mouse_press(self, event):
        try:
            # if
            # Get the position of the mouse click
            x = event.pos().x()
            y = event.pos().y()

            # Get the color of the pixel at the clicked position
            pixmap = self.ui.phantom_image_label.pixmap()
            if pixmap is not None:
                pixel_color = pixmap.toImage().pixel(x, y)
                intensity = QColor(pixel_color).getRgb()[0]
                index = np.where(
                    self.most_frequent == intensity)[0][0]
                t1 = self.t1Weight[index]
                t2 = self.t2Weight[index]
                pd = self.PDWeight[index]
                self.ui.label_4.setText(str(t1))
                self.ui.label_5.setText(str(t2))
                self.ui.label_6.setText(str(pd))

                # Remove the previous rectangle, if any
                painter = QPainter(pixmap)
                pen = QPen()
                pen.setColor(QColor(0, 255, 255))
                painter.setPen(pen)
                painter.setBrush(QBrush(QtCore.Qt.NoBrush))
                # Use the previously saved rectangle
                if self.prev_x != 0 and self.prev_y != 0:
                    painter.drawRect(
                        QRect(self.prev_x-5, self.prev_y-5, 10, 10))

                # Draw a rectangle around the selected pixel
                pen.setColor(QColor(255, 0, 0))
                painter.setPen(QPen(QtCore.Qt.red))
                painter.setBrush(QBrush(QtCore.Qt.NoBrush))
                self.rect = QRect(x-5, y-5, 10, 10)  # Save the new rectangle
                painter.drawRect(self.rect)  # Draw the new rectangle
                self.ui.phantom_image_label.setPixmap(pixmap)

                self.prev_x = x
                self.prev_y = y

        except Exception as e:
            print(e)

    def modify_the_image_intensities_distribution(self):
        try:
            img = cv2.imread(
                './images/shepp_logan_phantom/480px-Shepp_logan.png', 0)

            pixels = img.flatten()

            count = Counter(pixels)

            self.most_frequent = heapq.nlargest(10, count, key=count.get)

            self.most_frequent = np.sort(self.most_frequent)

            np.savetxt('./txt_files/most_frequent.txt',
                       np.sort(self.most_frequent), fmt='%d')

            for i in range(256):
                if i not in self.most_frequent:
                    nearest = min(self.most_frequent, key=lambda x: abs(x - i))
                    pixels[pixels == i] = nearest

            modified_img = pixels.reshape(img.shape)

            cv2.imwrite(
                './images/phantom_modified/480px-Shepp_logan.png', modified_img)

            self.t1WeightedImage = np.zeros_like(modified_img)
            self.t2WeightedImage = np.zeros_like(modified_img)
            self.PDWeightedImage = np.zeros_like(modified_img)

            self.t1Weight = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250]
            self.t2Weight = [250, 225, 200, 175, 150, 125, 100, 75, 50, 25]
            self.PDWeight = [255, 255, 255, 255, 255, 255, 255, 255, 255, 255]

            for i, intensity in enumerate(self.most_frequent):
                self.t1WeightedImage[modified_img ==
                                     intensity] = self.t1Weight[i]
                self.t2WeightedImage[modified_img ==
                                     intensity] = self.t2Weight[i]
                self.PDWeightedImage[modified_img ==
                                     intensity] = self.PDWeight[i]

            # Save the modified image and the three unique intensity images locally
            cv2.imwrite('./images/features_images/t1WeightedImage.png',
                        self.t1WeightedImage)
            cv2.imwrite('./images/features_images/t2WeightedImage.png',
                        self.t2WeightedImage)
            cv2.imwrite('./images/features_images/PDWeightedImage.png',
                        self.PDWeightedImage)

        except Exception as e:
            print(e)

    def handle_image_features_combo_box(self):
        try:
            if self.ui.comboBox.currentIndex() == 0:
                self.show_image_on_label(
                    './images/phantom_modified/480px-Shepp_logan.png')
            elif self.ui.comboBox.currentIndex() == 1:
                self.show_image_on_label(
                    './images/features_images/t1WeightedImage.png')
            elif self.ui.comboBox.currentIndex() == 2:
                self.show_image_on_label(
                    './images/features_images/t2WeightedImage.png')
            elif self.ui.comboBox.currentIndex() == 3:
                self.show_image_on_label(
                    './images/features_images/PDWeightedImage.png')
        except Exception as e:
            print(e)

    # MRI Sequence
    def apply_rf_pulse_on_image(self):
        try:
            # img = self.original_phantom_image.copy()
            img = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
            rows, columns = img.shape

            # define the phase of the gradient in x and y direction
            self.gx_phases = np.arange(0, 360, 360 / rows)
            self.gy_phases = np.arange(0, 360, 360 / rows)
            self.gy_counter = 0
            self.kspace = np.zeros((rows, columns, 3))
            new_3D_matrix_image = np.zeros((rows, columns, 3))
            # make a rotation matrix with 90 along x-axis
            theta = np.pi / 2  # angle in radians
            rotation_matrix = np.array([[1, 0, 0],
                                        [0, np.cos(theta), -np.sin(theta)],
                                        [0, np.sin(theta), np.cos(theta)]])
            # loop over each pixel in the image
            for i in range(rows):
                for j in range(columns):
                    # define the vector Mo
                    Mo = [0, 0, img[i, j]]
                    # Multiply the vector v by the  rotation_matrix to get the flipped vector v_flipped
                    Mo_flipped_xy_plane = np.round(
                        np.dot(rotation_matrix, Mo), 2)
                    new_3D_matrix_image[i, j] = Mo_flipped_xy_plane

            print(new_3D_matrix_image)
            return new_3D_matrix_image
        except Exception as e:
            print(e)

    def apply_phase_encoding_Gy_gradient(self, image_3d_matrix):
        try:
            # angle in radians
            theta = self.gy_phases[self.gy_counter] * np.pi / 180

            # make a 3d rotation_matrix in z direction
            rotation_matrix = np.array([[np.cos(theta), np.sin(theta), 0],
                                        [-np.sin(theta), np.cos(theta), 0],
                                        [0, 0, 1]])

            rows, columns = image_3d_matrix.shape[:2]
            # loop over each pixel in the image
            for i in range(rows):
                for j in range(columns):
                    # define the vector Mo
                    Mo = image_3d_matrix[i, j]
                    # Multiply the vector v by the rotation matrix R to get the flipped vector v_flipped
                    Mo_flipped_xy_plane = np.round(
                        np.dot(rotation_matrix, Mo), 2)

                    image_3d_matrix[i, j] = Mo_flipped_xy_plane

            self.gy_counter += 1
            print(image_3d_matrix)
            return image_3d_matrix
        except Exception as e:
            print(e)

    def apply_freqency_encoding_Gx_gradient(self, image_3d_matrix_after_gy):
        try:
            # make a copy of the image

            for index, phase in enumerate(self.gx_phases):
                image_3d_matrix_copy = image_3d_matrix_after_gy.copy()
                # angle in radians
                theta = phase * np.pi / 180
                # make a 3d rotation_matrix in z direction
                rotation_matrix = np.array([[np.cos(theta), np.sin(theta), 0],
                                            [-np.sin(theta), np.cos(theta), 0],
                                            [0, 0, 1]])

                rows, columns = image_3d_matrix_after_gy.shape[:2]
                # loop over each pixel in the image
                for i in range(rows):
                    for j in range(columns):
                        # define the vector Mo
                        Mo = image_3d_matrix_after_gy[i, j]
                        # Multiply the vector v by the rotation matrix R to get the flipped vector v_flipped
                        Mo_flipped_xy_plane = np.round(
                            np.dot(rotation_matrix, Mo), 2)

                        image_3d_matrix_copy[i, j] = Mo_flipped_xy_plane

                self.kspace[self.gy_counter, index] = np.sum(
                    image_3d_matrix_copy)
                print(
                    f"row: {self.gy_counter}, column: {index}, kspace: {self.kspace[self.gy_counter, index]}")
        except Exception as e:
            print(e)

    def read_out_signal(self):
        try:
            for i in range(self.kspace.shape[0]):
                for j in range(self.kspace.shape[1]):
                    magnitude = self.kspace[i, j].real
                    phase = self.kspace[i, j].imag
        except Exception as e:
            print(e)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())
