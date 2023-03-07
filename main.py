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

        self.ui.comboBox.currentIndexChanged.connect(
            self.handle_image_features_combo_box)
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
        self.show_image_on_label(
            './images/phantom_modified/480px-Shepp_logan.png')
        self.modify_the_image_intensities_distribution()
        self.ui.phantom_image_label.mousePressEvent = self.handle_mouse_press

    @QtCore.pyqtSlot()
    def show_image_on_label(self, image_path):
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
                index = np.where(
                    self.most_frequent == intensity)[0][0]
                t1 = self.t1Weight[index]
                t2 = self.t2Weight[index]
                pd = self.PDWeight[index]
                self.ui.label_4.setText(str(t1))
                self.ui.label_5.setText(str(t2))
                self.ui.label_6.setText(str(pd))

        except Exception as e:
            print(e)

    def modify_the_image_intensities_distribution(self):

        img = cv2.imread('./images/480px-Shepp_logan.png', 0)

        pixels = img.flatten()

        count = Counter(pixels)

        self.most_frequent = heapq.nlargest(10, count, key=count.get)

        self.most_frequent = np.sort(self.most_frequent)

        np.savetxt('./txt_files/most_frequent.txt',
                   np.sort(self.most_frequent), fmt='%d')
        for i in self.most_frequent:
            print(f"Pixel intensity {i}: {count[i]}")

        for i in range(256):
            if i not in self.most_frequent:
                nearest = min(self.most_frequent, key=lambda x: abs(x-i))
                pixels[pixels == i] = nearest

        modified_img = pixels.reshape(img.shape)

        cv2.imwrite(
            './images/phantom_modified/480px-Shepp_logan.png', modified_img)

        self.t1WeightedImage = np.zeros_like(modified_img)
        self.t2WeightedImage = np.zeros_like(modified_img)
        self.PDWeightedImage = np.zeros_like(modified_img)

        self.t1Weight = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250]
        self.t2Weight = [250, 225, 200, 175, 150, 125, 100, 75, 50, 25]
        self. PDWeight = [255, 255, 255, 255, 255, 255, 255, 255, 255, 255]

        for i, intensity in enumerate(self.most_frequent):
            self.t1WeightedImage[modified_img == intensity] = self.t1Weight[i]
            self.t2WeightedImage[modified_img == intensity] = self.t2Weight[i]
            self.PDWeightedImage[modified_img == intensity] = self.PDWeight[i]

        # Save the modified image and the three unique intensity images locally
        cv2.imwrite('./images/features_images/t1WeightedImage.png',
                    self.t1WeightedImage)
        cv2.imwrite('./images/features_images/t2WeightedImage.png',
                    self.t2WeightedImage)
        cv2.imwrite('./images/features_images/PDWeightedImage.png',
                    self.PDWeightedImage)

    def handle_image_features_combo_box(self):
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
