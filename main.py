from collections import Counter
import json
import math
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
        self.phantom_image_resized = cv2.imread(
            "./images/phantom_modified/16x16.png", 0)
        self.image_path = './images/shepp_logan_phantom/64x64.png'
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
        self.ui.pushButton.clicked.connect(
            lambda: self.apply_rf_pulse(self.phantom_image_resized, 90))
        self.ui.pushButton_2.clicked.connect(
            lambda: self.apply_gradient(self.new_3D_matrix_image))

        # self.ui.pushButton.clicked.connect(self.update_image)
        self.ui.updateBtn.released.connect(self.update)

        # MRI Sequence

        # run the function to read the starting values from the dials
        self.read_dial_values_and_calculate_ernst()
        # connect the dials to the function
        self.ui.dial.valueChanged.connect(
            self.read_dial_values_and_calculate_ernst)
        self.ui.dial_2.valueChanged.connect(
            self.read_dial_values_and_calculate_ernst)

    def browseFile(self):
        # get jason file data and store it in a variable
        self.fileName = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Open File', './', 'Json Files (*.json)')

        self.update()

    def update(self):
        # get the values from the dictionary
        self.data = json.load(open(self.fileName[0]))
        self.plot_Rf(*self.data['RF'])
        self.plot_Gss(*self.data['GSS'])
        self.plot_Gpe(*self.data['GPE'])
        self.plot_Gfe(*self.data['GFE'])
        self.plot_RO(*self.data['RO'])

    def prebGraphData(self, start, amp, num=1, function=half_sin_wave, repPerPlace=1, elevation=0, step=1, oscillation=False):
        yAxiesVal = []
        xAxiesVal = []

        for j in range(int(num)):
            for i in np.linspace(1, -1, repPerPlace):
                yAxiesVal.extend(elevation + (function(amp) * i * np.power(-1, j))
                                 if oscillation else elevation + (function(amp) * i))
                xAxiesVal.extend(np.arange(start, start + 1, 1/100))
            start += step

        return [xAxiesVal, yAxiesVal]

    def plot_Rf(self, start, amp, num=1):
        xAxiesVal, yAxiesVal = self.prebGraphData(
            start, amp, num, half_sin_wave, elevation=10, oscillation=True)
        self.RFplotter.setData(xAxiesVal, yAxiesVal)

    def plot_Gss(self, start, amp, num=1):
        xAxiesVal, yAxiesVal = self.prebGraphData(
            start, amp, num, square_wave, elevation=7.5, step=1)
        self.GSSplotter.setData(xAxiesVal, yAxiesVal)

    def plot_Gpe(self, start, amp, num=1):
        xAxiesVal, yAxiesVal = self.prebGraphData(
            start, amp, num, square_wave, repPerPlace=5, elevation=5, step=2)
        self.GPEplotter.setData(xAxiesVal, yAxiesVal)

    def plot_Gfe(self, start, amp, num=1):
        xAxiesVal, yAxiesVal = self.prebGraphData(
            start, amp, num, square_wave, elevation=2.5, step=1)
        self.GFEplotter.setData(xAxiesVal, yAxiesVal)

    def plot_RO(self, start, amp, num=1):
        xAxiesVal, yAxiesVal = self.prebGraphData(
            start, amp, num, half_sin_wave)
        self.ROplotter.setData(xAxiesVal, yAxiesVal)

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

            # img_min = img.min()
            # img_max = img.max()
            # img = (img - img_min) * 255.0 / (img_max - img_min)

            # # Ensure that the values of img are between 0 and 255
            # img = np.clip(img, 0, 255)

            indices = np.where(img > 255)

            img[indices] = np.minimum(img[indices], 255)

            indices = np.where(img < 0)

            img[indices] = np.maximum(img[indices], 0)
            # Convert the data type of img to uint8 (unsigned 8-bit integer)
            np.savetxt('IMAGE.txt',
                       img, fmt='%d')
            img = img.astype(np.uint8)

            # # # Invert the colors if necessary
            # if img_max > 255:
            #     img = 255 - img

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

    def read_dial_values_and_calculate_ernst(self):
        try:
            tr_value = int(self.ui.dial.value())
            t1_value = int(self.ui.dial_2.value())
            ernst_angle = int(np.arccos(
                math.exp(-tr_value/t1_value)) * (180.0 / math.pi))
            self.ui.label_8.setText(str(self.ui.dial.value()))
            self.ui.label_9.setText(str(self.ui.dial_2.value()))
            self.ui.label_13.setText(str(ernst_angle))
        except Exception as e:
            print(e)

    def apply_rf_pulse(self, image, flip_angle):
        rows, columns = image.shape
        # make a rotation matrix with 90 along x-axis
        theta = flip_angle * np.pi/180  # angle in radians
        self.gx_phases = np.arange(0, 360, 360 / rows)
        self.gy_phases = np.arange(0, 360, 360 / rows)
        # rotation along y axis
        rotation_matrix_in_y = np.array([[np.cos(theta), 0, np.sin(theta)],
                                         [0, 1, 0],
                                         [-np.sin(theta), 0, np.cos(theta)]])

        self.new_3D_matrix_image = np.zeros((rows, columns, 3))
        # loop over each pixel in the image
        for i in range(rows):
            for j in range(columns):
                # define the vector Mo
                Mo = [0, 0, image[i, j]]
                # Multiply the vector v by the rotation matrix rotation_matrix_in_y to get the flipped vector v_flipped
                Mo_flipped_xy_plane = np.round(
                    np.dot(rotation_matrix_in_y, Mo), 2)
                self.new_3D_matrix_image[i, j] = Mo_flipped_xy_plane

    def apply_gradient(self, image_after_rf_pulse):
        backup_image = image_after_rf_pulse.copy()
        rows, columns, _ = image_after_rf_pulse.shape
        k_space_2d = np.zeros((rows, columns), dtype=complex)
        k_space = np.ones((rows, columns))
        phases = np.zeros((rows, columns))

        for row_index, gy_phase in enumerate(self.gy_phases):
            for column_index, gx_phase in enumerate(self.gx_phases):
                image_after_rf_pulse = backup_image.copy()
                for i in range(rows):
                    for j in range(columns):
                        pixel_value = image_after_rf_pulse[i, j, 0]
                        phase_from_gy = gy_phase * i
                        phase_from_gx = gx_phase * j
                        applied_phase = (phase_from_gx + phase_from_gy)
                        phases[i, j] = applied_phase
                        applied_phase *= np.pi/180
                        new_x_value = pixel_value * np.cos(applied_phase)
                        new_y_value = pixel_value * np.sin(applied_phase)
                        image_after_rf_pulse[i, j, 0] = new_x_value
                        image_after_rf_pulse[i, j, 1] = new_y_value
                sum = np.round(np.sum(image_after_rf_pulse, axis=(0, 1)), 2)
                k_space_2d[row_index][column_index] = np.round(
                    sum[0], 2) - 1j * np.round(sum[1], 2)

                k_space[row_index, column_index] = np.sqrt(
                    sum[0]**2 + sum[1]**2)

                # self.update_kspace(k_space)
        self.update_image(k_space_2d)

    def update_kspace(self, kspace):

        resized = cv2.resize(kspace, (300, 300))
        img = QImage(
            resized.tobytes(), resized.shape[1], resized.shape[0], QImage.Format_Grayscale8)

        pixmap = QPixmap.fromImage(img)
        self.ui.k_space_label.setPixmap(pixmap)
        self.ui.k_space_label.setMaximumSize(300, 300)
        self.ui.k_space_label.setMinimumSize(300, 300)

    def update_image(self, kspace_2d):
        # image = cv2.imread(
        #     './images/phantom_modified/16x16.png', cv2.IMREAD_GRAYSCALE)

        # kspace_2d_1 = np.fft.fft2(image)
        # img = np.fft.ifft2(kspace_2d_1)
        img = np.fft.ifft2(kspace_2d)
        img = np.real(img).astype(np.uint8)
        # cv2.imwrite('./images/features_images/kspace_2d.png', img)
        # arr2 = cv2.imread('./images/features_images/kspace_2d.png')
        # height, width, _ = kspace_2d_1.shape

        # img = QImage(arr2, width=width, height=height,
        #              format=QImage.Format_Grayscale8)
        resized = cv2.resize(img, (300, 300))
        new_img = QImage(resized.tobytes(), resized.shape[1], resized.shape[0],
                         QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(new_img)
        self.ui.reconstructed_image_label.setPixmap(pixmap)
        self.ui.reconstructed_image_label.setMaximumSize(300, 300)
        self.ui.reconstructed_image_label.setMinimumSize(300, 300)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())
