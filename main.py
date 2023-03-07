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

        # phantom
        # self.modify_the_image_intensities_distribution()
        self.show_image_on_label()
        # self.modify_the_image_intensities_distribution()
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
                print("Pixel intensity:", intensity)
        except Exception as e:
            print(e)

    def modify_the_image_intensities_distribution(self):

        img = cv2.imread('./images/480px-Shepp_logan.png', 0)

        # Flatten the image array
        pixels = img.flatten()

        # Count the occurrences of each pixel intensity
        count = Counter(pixels)

        # Find the top five most frequent pixel intensities
        most_frequent = heapq.nlargest(10, count, key=count.get)

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

        return modified_img

    def show_image_histogram(self):
        try:

            # Load the image
            image = plt.imread('./images/480px-Shepp_logan.png')

            # Compute the histogram
            hist, bins = np.histogram(image.ravel(), bins=256, range=(0, 255))

            # Find the peak value
            peak_value = bins[np.argmax(hist)]
            print(bins)
            print("Peak value:", peak_value)
            # Create a new figure
            fig, ax = plt.subplots()

            # Display the new image
            ax.imshow(image, cmap='gray')

            # Show the plot
            plt.show()
        except Exception as e:
            print(e)

    def get_image_intensities(self):
        image = self.ui.phantom_image_label.pixmap().toImage()
        intensities = []
        for y in range(image.height()):
            for x in range(image.width()):
                pixel_color = image.pixel(x, y)
                intensity = QColor(pixel_color).getRgb()[0]
                intensities.append(intensity)
        return intensities

    def get_unique_image_intensities(self):
        # Get the unique intensities in the image
        image = self.ui.phantom_image_label.pixmap().toImage()
        intensities = set()
        for y in range(image.height()):
            for x in range(image.width()):
                pixel_color = image.pixel(x, y)
                intensity = QColor(pixel_color).getRgb()[0]
                intensities.add(intensity)

        print("Unique intensities:", intensities)

    def create_the_corresponding_matrices(self, height, width):
        # Create a matrix with the same shape as the loaded image
        self.T1Matrix = np.zeros((height, width))
        self.T2Matrix = np.zeros((height, width))
        self.PDMatrix = np.zeros((height, width))

        # Assign specific values at each intensity in the matrix
        intensities = [0, 101, 76, 51, 25, 255]
        t1matrix = [0, 50, 100, 150, 200, 255]
        t2matrix = [255, 200, 150, 100, 50, 0]
        pdmatrix = [255, 255, 255, 255, 255, 255]

        for intensity, t1, t2, pd in zip(intensities, t1matrix, t2matrix, pdmatrix):
            for y in range(height):
                for x in range(width):
                    pixel_intensity = QtGui.qRed(self.image.pixel(x, y))
                    if pixel_intensity == intensity:
                        self.T1Matrix[x, y] = t1
                        self.T2Matrix[x, y] = t2
                        self.PDMatrix[x, y] = pd

        self.create_the_corresponding_images()
        # write each matrix in a txt file
        np.savetxt('T1Matrix.txt', self.T1Matrix, fmt='%d')
        np.savetxt('T2Matrix.txt', self.T2Matrix, fmt='%d')
        np.savetxt('PDMatrix.txt', self.PDMatrix, fmt='%d')

    def create_the_corresponding_images(self):
        # The issue you are experiencing might be related to the difference in how OpenCV and Qt handle image orientation. OpenCV uses the BGR color format by default, while Qt uses the RGB format. This can cause the image to appear mirrored.

        self.T1Matrix = cv2.flip(self.T1Matrix, 1)
        self.T1Matrix = cv2.rotate(
            self.T1Matrix, cv2.ROTATE_90_COUNTERCLOCKWISE)  # Rotate clockwise
        cv2.imwrite('./images/T1Matrix.jpg', self.T1Matrix)

        self.T2Matrix = cv2.flip(self.T2Matrix, 1)
        self.T2Matrix = cv2.rotate(
            self.T2Matrix, cv2.ROTATE_90_COUNTERCLOCKWISE)  # Rotate clockwise
        cv2.imwrite('./images/T2Matrix.jpg', self.T2Matrix)

        self.PDMatrix = cv2.flip(self.PDMatrix, 1)
        self.PDMatrix = cv2.rotate(
            self.PDMatrix, cv2.ROTATE_90_COUNTERCLOCKWISE)  # Rotate clockwise
        cv2.imwrite('./images/PDMatrix.jpg', self.PDMatrix)

    def load_matrices_from_images(self):
        self.T1Matrix = cv2.imread('./images/T1Matrix.jpg', 0)
        self.T2Matrix = cv2.imread('./images/T2Matrix.jpg', 0)
        self.PDMatrix = cv2.imread('./images/PDMatrix.jpg', 0)

    def handle_combo_box(self):
        if self.ui.comboBox.currentIndex() == 0:
            self.image = cv2.imread('./images/Shepp_logan.png')
            height, width, channel = self.image.shape
            self.heightttt = height
            bytesPerLine = 3 * width
            self.image = QtGui.QImage(
                self.image.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888).rgbSwapped()
            self.ui.phantom_image_label.setPixmap(
                QtGui.QPixmap.fromImage(self.image))
            self.ui.phantom_image_label.setScaledContents(True)
        elif self.ui.comboBox.currentIndex() == 1:
            self.T1Matrix = cv2.imread('./images/T1Matrix.jpg')
            height, width, channel = self.T1Matrix.shape
            self.heightttt = height
            bytesPerLine = 3 * width
            self.T1Matrix = QtGui.QImage(
                self.T1Matrix.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888).rgbSwapped()
            self.ui.phantom_image_label.setPixmap(
                QtGui.QPixmap.fromImage(self.T1Matrix))
            self.ui.phantom_image_label.setScaledContents(True)

        elif self.ui.comboBox.currentIndex() == 2:
            self.T2Matrix = cv2.imread('./images/T2Matrix.jpg')
            height, width, channel = self.T2Matrix.shape
            self.heightttt = height
            bytesPerLine = 3 * width
            self.T2Matrix = QtGui.QImage(
                self.T2Matrix.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888).rgbSwapped()
            self.ui.phantom_image_label.setPixmap(
                QtGui.QPixmap.fromImage(self.T2Matrix))
            self.ui.phantom_image_label.setScaledContents(True)
        elif self.ui.comboBox.currentIndex() == 3:
            self.PDMatrix = cv2.imread('./images/PDMatrix.jpg')
            height, width, channel = self.PDMatrix.shape
            self.heightttt = height
            bytesPerLine = 3 * width
            self.PDMatrix = QtGui.QImage(
                self.PDMatrix.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888).rgbSwapped()
            self.ui.phantom_image_label.setPixmap(
                QtGui.QPixmap.fromImage(self.PDMatrix))
            self.ui.phantom_image_label.setScaledContents(True)

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


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())
