from collections import Counter
import json
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QPixmap, QImage, QColor, QPainter, QPen, QBrush
from PyQt5.QtCore import Qt, QRect
import cv2
import sys
import numpy as np
import pyqtgraph as pg
import heapq
from ui_mainWindow import Ui_MainWindow


def square_wave(Amp, NumOfPoints=100):
    arr = np.full(NumOfPoints, Amp)
    arr[0], arr[-1] = 0, 0
    return arr


def half_sin_wave(Amp, Freq=1):
    t_sin = np.arange(0, 1, 1/100)
    return Amp * np.sin(np.pi * Freq * t_sin)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # phantom image
        self.prev_x = 0
        self.prev_y = 0
        self.image_path = './images/shepp_logan_phantom/480px-Shepp_logan.png'
        self.ui.comboBox_2.currentIndexChanged.connect(
            self.change_phantom_size)
        self.ui.comboBox.currentIndexChanged.connect(
            self.handle_image_features_combo_box)
        self.modify_the_image_intensities_distribution(self.image_path)
        self.show_image_on_label(self.image_path)
        self.ui.phantom_image_label.mousePressEvent = self.handle_mouse_press
        self.brightness = 100
        self.ui.phantom_image_label.wheelEvent = self.handle_wheel_event

        self.redPen = pg.mkPen(color=(255, 0, 0), width=2)
        self.greenPen = pg.mkPen(color=(0, 255, 0), width=2)
        self.bluePen = pg.mkPen(color=(0, 0, 255), width=2)
        self.whitePen = pg.mkPen(color=(255, 255, 255))

        self.ui.signalPlot.setXRange(-0.5, 10.5, padding=0)
        self.ui.signalPlot.setYRange(-1, 11.5, padding=0)

        self.ui.signalPlot.plotItem.addLine(y=0, pen=self.whitePen)
        self.ui.signalPlot.plotItem.addLine(y=2.5, pen=self.whitePen)
        self.ui.signalPlot.plotItem.addLine(y=5, pen=self.whitePen)
        self.ui.signalPlot.plotItem.addLine(y=7.5, pen=self.whitePen)
        self.ui.signalPlot.plotItem.addLine(y=10, pen=self.whitePen)

        self.ui.signalPlot.plotItem.hideAxis('left')

        self.RFplotter = self.ui.signalPlot.plot([], [], pen=self.redPen)
        self.GSSplotter = self.ui.signalPlot.plot([], [], pen=self.greenPen)
        self.GPEplotter = self.ui.signalPlot.plot([], [], pen=self.bluePen)
        self.GFEplotter = self.ui.signalPlot.plot([], [], pen=self.redPen)
        self.ROplotter = self.ui.signalPlot.plot([], [], pen=self.greenPen)

        self.ui.browseFileBtn.released.connect(self.browseFile)
        self.ui.updateBtn.released.connect(self.update)

        # MRI Sequence

    @QtCore.pyqtSlot()
    def show_image_on_label(self, image_path):
        try:
            self.original_phantom_image = cv2.imread(image_path, 0)
            # modify the label size to fit the image
            self.ui.phantom_image_label.setMaximumSize(
                self.original_phantom_image.shape[1], self.original_phantom_image.shape[0])
            self.ui.phantom_image_label.setMinimumSize(
                self.original_phantom_image.shape[1], self.original_phantom_image.shape[0])

            self.mean, self.std_dev = cv2.meanStdDev(
                self.original_phantom_image)
            img = QImage(image_path)
            pixmap = QPixmap.fromImage(img)
            self.ui.phantom_image_label.setPixmap(pixmap)
        except Exception as e:
            print(e)

    def handle_wheel_event(self, event):
        try:
            delta = event.angleDelta().y()
            if delta > 0:
                self.brightness += 10
            else:
                self.brightness -= 10

            # Set threshold for all white image
            if self.brightness > 245:
                self.brightness = 255
            elif self.brightness < -245:
                self.brightness = -255

            img = self.original_phantom_image + self.brightness
            np.savetxt('IMAGE.txt',
                       img, fmt='%d')

            img_min = img.min()
            img_max = img.max()
            img = (img - img_min) * 255.0 / (img_max - img_min)

            # Ensure that the values of img are between 0 and 255
            img = np.clip(img, 0, 255)

            # Convert the data type of img to uint8 (unsigned 8-bit integer)
            img = img.astype(np.uint8)

            # # Invert the colors if necessary
            if img_max > 255:
                img = 255 - img

            # self.modify_the_image_intensities_distribution(self.image_path, img)

            qtImg = QImage(img, img.shape[1], img.shape[0], img.strides[0],
                           QImage.Format_Grayscale8)
            self.ui.phantom_image_label.setPixmap(QPixmap.fromImage(qtImg))
        except Exception as e:
            print(e)

    def handle_mouse_press(self, event):
        try:
            if self.ui.comboBox.currentText() == 'Image':
                self.show_image_on_label(self.image_path)
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
                    # Draw a rectangle around the selected pixel
                    painter.setPen(QPen(QtCore.Qt.red))
                    painter.setBrush(QBrush(QtCore.Qt.NoBrush))
                    # Save the new rectangle
                    self.rect = QRect(x-5, y-5, 10, 10)
                    painter.drawRect(self.rect)  # Draw the new rectangle
                    self.ui.phantom_image_label.setPixmap(pixmap)

        except Exception as e:
            print(e)

    def modify_the_image_intensities_distribution(self, img_path='./images/shepp_logan_phantom/480px-Shepp_logan.png', image=None):
        try:
            if image is not None:
                img = image
            else:
                img = cv2.imread(
                    img_path, 0)

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
                './images/phantom_modified/'+str(img_path.split('/')[-1]), modified_img)

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
                self.show_image_on_label(self.image_path)
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

    def change_phantom_size(self):
        try:
            index = self.ui.comboBox_2.currentIndex()
            if index == 0:
                return
            if index == 1:
                self.phantom_image_resized = cv2.resize(
                    self.original_phantom_image, (16, 16))

            elif index == 2:
                self.phantom_image_resized = cv2.resize(
                    self.original_phantom_image, (32, 32))

            elif index == 3:
                self.phantom_image_resized = cv2.resize(
                    self.original_phantom_image, (64, 64))

            self.modify_the_image_intensities_distribution(self.image_path)
            self.show_image_on_label(self.image_path)
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

    def browseFile(self):
        try:
            # get jason file data and store it in a variable
            self.fileName = QtWidgets.QFileDialog.getOpenFileName(
                self, 'Open File', './', 'Json Files (*.json)')

            self.update()
        except Exception as e:
            print(e)

    def update(self):
        try:
            # get the values from the dictionary
            self.data = json.load(open(self.fileName[0]))
            self.plot_Rf(*self.data['RF'])
            self.plot_Gss(*self.data['GSS'])
            self.plot_Gpe(*self.data['GPE'])
            self.plot_Gfe(*self.data['GFE'])
            self.plot_RO(*self.data['RO'])
        except Exception as e:
            print(e)

    def prebGraphData(self, start, amp, num=1, function=half_sin_wave, repPerPlace=1, elevation=0, step=1, oscillation=False):
        try:
            yAxiesVal = []
            xAxiesVal = []

            for j in range(int(num)):
                for i in np.linspace(1, -1, repPerPlace):
                    yAxiesVal.extend(elevation + (function(amp) * i * np.power(-1, j))
                                     if oscillation else elevation + (function(amp) * i))
                    xAxiesVal.extend(np.arange(start, start + 1, 1/100))
                start += step

            return [xAxiesVal, yAxiesVal]
        except Exception as e:
            print(e)

    def plot_Rf(self, start, amp, num=1):
        try:
            xAxiesVal, yAxiesVal = self.prebGraphData(
                start, amp, num, half_sin_wave, elevation=10, oscillation=True)
            self.RFplotter.setData(xAxiesVal, yAxiesVal)
        except Exception as e:
            print(e)

    def plot_Gss(self, start, amp, num=1):
        try:
            xAxiesVal, yAxiesVal = self.prebGraphData(
                start, amp, num, square_wave, elevation=7.5, step=1)
            self.GSSplotter.setData(xAxiesVal, yAxiesVal)
        except Exception as e:
            print(e)

    def plot_Gpe(self, start, amp, num=1):
        try:
            xAxiesVal, yAxiesVal = self.prebGraphData(
                start, amp, num, square_wave, repPerPlace=5, elevation=5, step=2)
            self.GPEplotter.setData(xAxiesVal, yAxiesVal)
        except Exception as e:
            print(e)

    def plot_Gfe(self, start, amp, num=1):
        try:
            xAxiesVal, yAxiesVal = self.prebGraphData(
                start, amp, num, square_wave, elevation=2.5, step=1)
            self.GFEplotter.setData(xAxiesVal, yAxiesVal)
        except Exception as e:
            print(e)

    def plot_RO(self, start, amp, num=1):
        try:
            xAxiesVal, yAxiesVal = self.prebGraphData(
                start, amp, num, half_sin_wave)
            self.ROplotter.setData(xAxiesVal, yAxiesVal)
        except Exception as e:
            print(e)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())
