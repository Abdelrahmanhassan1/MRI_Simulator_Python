from collections import Counter
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtGui import QPixmap, QImage, QColor
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

        self.ui.comboBox.currentIndexChanged.connect(self.handle_combo_box)
        self.plot_horizontal_lines()
        self.redPen = pg.mkPen(color=(255, 0, 0))  # RED
        self.greenPen = pg.mkPen(color=(0, 255, 0))  # Green
        self.bluePen = pg.mkPen(color=(0, 0, 255))  # BLue
        self.blackPen = pg.mkPen(color=(0, 0, 0))  # Black
        self.orangePen = pg.mkPen(color=(255, 165, 0))  # Orange

        self.pens = [
            self.redPen,
            self.greenPen,
            self.bluePen,
            self.blackPen,
            self.orangePen
        ]

        # phantom image
        self.show_image_on_label()
        self.modify_the_image_intensities_distribution()
        self.ui.phantom_image_label.mousePressEvent = self.handle_mouse_press
        # self.show_image_histogram()
        # self.get_unique_image_intensities()

    @QtCore.pyqtSlot()
    def show_image_on_label(self):
        try:
            self.shepp_logan_image = QImage(
                './images/phantom_modified/480px-Shepp_logan.png')
            pixmap = QPixmap.fromImage(self.shepp_logan_image)
            self.ui.phantom_image_label.setPixmap(pixmap)
            # self.get_unique_image_intensities()
        except Exception as e:
            print(e)

    def show_image(self, image_path):
        img = QImage(image_path)
        pixmap = QPixmap.fromImage(img)
        self.ui.phantom_image_label.setPixmap(pixmap)

    def handle_mouse_press(self, event):
        try:
            # Get the position of the mouse click
            x = event.pos().x()
            y = event.pos().y()

            # Get the color of the pixel at the clicked position
            pixmap = self.ui.phantom_image_label.pixmap()
            if pixmap is not None:
                pixel_color = pixmap.toImage().pixel(x, y)
                intensity = QColor(pixel_color).getRgb()[0]
                t1 = self.T1Matrix[x, y]
                t2 = self.T2Matrix[x, y]
                pd = self.PDMatrix[x, y]
                self.ui.label_4.setText(str(t1))
                self.ui.label_5.setText(str(t2))
                self.ui.label_6.setText(str(pd))

        except Exception as e:
            print(e)

    def modify_the_image_intensities_distribution(self):

        img = cv2.imread('./images/480px-Shepp_logan.png', 0)

        height, width = img.shape

        # Flatten the image array
        pixels = img.flatten()

        # Count the occurrences of each pixel intensity
        count = Counter(pixels)

        # Find the top five most frequent pixel intensities
        most_frequent = heapq.nlargest(10, count, key=count.get)

        # save the most_frequent intensities in a txt file
        np.savetxt('./txt_files/most_frequent.txt',
                   np.sort(most_frequent), fmt='%d')

        # Print the top five most frequent pixel intensities and their counts
        for i in most_frequent:
            print(f"Pixel intensity {i}: {count[i]}")

        # Overwrite the image array to have the nearest value of the top five most frequent pixel intensities
        for i in range(256):
            if i not in most_frequent:
                nearest = min(most_frequent, key=lambda x: abs(x-i))
                pixels[pixels == i] = nearest

        # Reshape the modified array to its original shape
        modified_img = pixels.reshape(img.shape)

        # save the image locally
        cv2.imwrite(
            './images/phantom_modified/480px-Shepp_logan.png', modified_img)

        self.create_the_corresponding_matrices(
            height=height, width=width, most_frequent_intensities=most_frequent, phantom_image=modified_img)

        return modified_img

    def create_the_corresponding_matrices(self, height, width, most_frequent_intensities, phantom_image):
        # Create a matrix with the same shape as the loaded image
        self.T1Matrix = np.zeros((height, width))
        self.T2Matrix = np.zeros((height, width))
        self.PDMatrix = np.zeros((height, width))

        # Assign specific values at each intensity in the matrix
        t1matrix = np.linspace(
            start=0, stop=255, num=len(most_frequent_intensities))
        t2matrix = np.linspace(
            start=255, stop=0, num=len(most_frequent_intensities))
        pdmatrix = np.repeat(255, len(most_frequent_intensities))

        for intensity, t1, t2, pd in zip(most_frequent_intensities, t1matrix, t2matrix, pdmatrix):
            for y in range(height):
                for x in range(width):
                    pixel_intensity = phantom_image[x, y]
                    if pixel_intensity == intensity:
                        self.T1Matrix[x, y] = t1
                        self.T2Matrix[x, y] = t2
                        self.PDMatrix[x, y] = pd

        self.create_the_corresponding_images()
        # write each matrix in a txt file
        np.savetxt('T1Matrix.txt', self.T1Matrix, fmt='%d')
        np.savetxt('T2Matrix.txt', self.T2Matrix, fmt='%d')
        np.savetxt('PDMatrix.txt', self.PDMatrix, fmt='%d')

        self.load_matrices_from_images()

    def create_the_corresponding_images(self):
        # The issue you are experiencing might be related to the difference in how OpenCV and Qt handle image orientation. OpenCV uses the BGR color format by default, while Qt uses the RGB format. This can cause the image to appear mirrored.

        # self.T1Matrix = cv2.flip(self.T1Matrix, 1)
        # self.T1Matrix = cv2.rotate(
        #     self.T1Matrix, cv2.ROTATE_90_COUNTERCLOCKWISE)  # Rotate clockwise
        cv2.imwrite('./images/features_images/T1Matrix.jpg', self.T1Matrix)

        # self.T2Matrix = cv2.flip(self.T2Matrix, 1)
        # self.T2Matrix = cv2.rotate(
        #     self.T2Matrix, cv2.ROTATE_90_COUNTERCLOCKWISE)  # Rotate clockwise
        cv2.imwrite('./images/features_images/T2Matrix.jpg', self.T2Matrix)

        # self.PDMatrix = cv2.flip(self.PDMatrix, 1)
        # self.PDMatrix = cv2.rotate(
        #     self.PDMatrix, cv2.ROTATE_90_COUNTERCLOCKWISE)  # Rotate clockwise
        cv2.imwrite('./images/features_images/PDMatrix.jpg', self.PDMatrix)

    def load_matrices_from_images(self):
        self.T1Matrix = cv2.imread('./images/features_images/T1Matrix.jpg', 0)
        self.T2Matrix = cv2.imread('./images/features_images/T2Matrix.jpg', 0)
        self.PDMatrix = cv2.imread('./images/features_images/PDMatrix.jpg', 0)

    def handle_combo_box(self):
        if self.ui.comboBox.currentIndex() == 0:
            self.show_image('./images/phantom_modified/480px-Shepp_logan.png')
        elif self.ui.comboBox.currentIndex() == 1:
            self.show_image('./images/features_images/T1Matrix.jpg')
        elif self.ui.comboBox.currentIndex() == 2:
            self.show_image('./images/features_images/T2Matrix.jpg')
        elif self.ui.comboBox.currentIndex() == 3:
            self.show_image('./images/features_images/PDMatrix.jpg')

    # MRI Sequence
    def plot_horizontal_lines(self):
        self.ui.graphicsView.plot(
            [0, 255], [0, 0], pen=pg.mkPen(color=(255, 0, 0)))
        self.draw_square_wave()
        self.ui.graphicsView.plot(
            [0, 255], [10, 10], pen=pg.mkPen(color=(255, 0, 0)))
        self.ui.graphicsView.plot(
            [0, 255], [20, 20], pen=pg.mkPen(color=(255, 0, 0)))
        self.ui.graphicsView.plot(
            [0, 255], [30, 30], pen=pg.mkPen(color=(255, 0, 0)))
        self.ui.graphicsView.plot(
            [0, 255], [40, 40], pen=pg.mkPen(color=(255, 0, 0)))

    def draw_sine_wave(self):
        x = np.linspace(0, 10, 101)
        y = np.sin(x)
        self.ui.graphicsView.plot(
            x, y, pen=pg.mkPen(color=(0, 0, 0)))

    def draw_square_wave(self):
        x = numpy.linspace(0, 20, 20)
        y = numpy.array([5 if math.floor(2 * t) % 2 == 0 else 0 for t in x])
        self.ui.graphicsView.plot(
            x, y, pen=pg.mkPen(color=(0, 0, 0)))


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())
