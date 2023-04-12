from collections import Counter
import json
import math
import time
from PyQt5.QtWidgets import QMessageBox
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QPixmap, QImage, QColor, QPainter, QPen, QBrush, QIntValidator, QFont
from PyQt5.QtCore import Qt, QRect
import cv2
import sys
import numpy as np
import pyqtgraph as pg
import heapq
import qimage2ndarray
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from ui_mainWindow import Ui_MainWindow

def flat_line(Amp, NumOfPoints=100):
    return np.full(NumOfPoints, Amp)

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
        # to control brightness of phantom
        self.setMouseTracking(True)
        self.dragging = False
        self.prev_pos = None
        self.brightness_factor = 0

        # kspace figure
        self.kspace_figure = Figure()
        self.kspace_canvas = FigureCanvas(self.kspace_figure)
        self.ui.verticalLayout_9.addWidget(self.kspace_canvas)

        # reconstructed image figure
        self.reconstructed_image_figure = Figure()
        self.reconstructed_image_canvas = FigureCanvas(
            self.reconstructed_image_figure)
        self.ui.verticalLayout_12.addWidget(self.reconstructed_image_canvas)

        self.ui.lineEdit.setValidator(QIntValidator())

        # phantom image
        self.prev_x = 0
        self.prev_y = 0
        self.new_3D_matrix_image = None
        self.scroll_flag = False
        self.phantom_image_resized = cv2.imread(
            "./images/phantom_modified/16x16.png", 0)

        self.image_path = './images/phantom_modified/480px-Shepp_logan.png'
        self.ui.comboBox_2.currentIndexChanged.connect(
            self.change_phantom_size)
        self.ui.comboBox.currentIndexChanged.connect(
            self.handle_image_features_combo_box)
        self.modify_the_image_intensities_distribution(self.image_path)
        self.show_image_on_label(self.image_path)
        self.ui.phantom_image_label.mousePressEvent = self.mousePressEvent
        self.ui.phantom_image_label.mouseReleaseEvent = self.mouseReleaseEvent
        self.ui.phantom_image_label.mousePressEvent = self.handle_mouse_press
        self.ui.phantom_image_label.mouseMoveEvent = self.mouseMoveEvent
        self.brightness = 0
        self.ui.phantom_image_label.wheelEvent = self.handle_wheel_event

        # MRI Sequence
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

        # set the y axis labels
        self.ui.signalPlot.plotItem.getAxis('left').setTicks(
            [((0, 'RO'), (2.5, 'Gx (FE)'), (5, 'Gy (PE)'), (7.5, 'Gz (SS)'), (10, 'RF'))])

        # set the x axis label
        self.ui.signalPlot.plotItem.getAxis('bottom').setLabel('Time (ms)')

        self.RFplotter = self.ui.signalPlot.plot([], [], pen=self.redPen)
        self.GSSplotter = self.ui.signalPlot.plot([], [], pen=self.greenPen)
        self.GPEplotter = self.ui.signalPlot.plot([], [], pen=self.bluePen)
        self.GFEplotter = self.ui.signalPlot.plot([], [], pen=self.redPen)
        self.ROplotter = self.ui.signalPlot.plot([], [], pen=self.greenPen)
        self.startingTR_plotter = self.ui.signalPlot.plot(
            [], [], pen=self.whitePen)
        self.TE_plotter = self.ui.signalPlot.plot([], [], pen=self.whitePen)
        self.TE_horizontal_line = self.ui.signalPlot.plot(
            [], [], pen=self.whitePen)
        self.endingTR_plotter = self.ui.signalPlot.plot(
            [], [], pen=self.whitePen)
        self.TR_horizontal_line = self.ui.signalPlot.plot(
            [], [], pen=self.whitePen)

        self.ui.browseFileBtn.released.connect(self.browseFile)
        self.ui.updateBtn.released.connect(self.update)

        # RF and Gradient
        self.ui.pushButton_2.released.connect(self.apply_rf_gradient_sequence)

        self.read_dial_values_and_calculate_ernst()
        for dial in [self.ui.dial, self.ui.dial_2, self.ui.dial_3]:
            dial.valueChanged.connect(
                self.read_dial_values_and_calculate_ernst)

        self.ui.horizontalSlider.valueChanged.connect(self.apply_noise)
        self.ui.pushButton_3.released.connect(self.reset_phantom_to_original)

        self.ui.pushButton.released.connect(self.plot_chosen_sequence)

    @ QtCore.pyqtSlot()
    def show_image_on_label(self, image_path, image=None):
        try:
            if image_path is not None:
                self.original_phantom_image = cv2.imread(image_path, 0)
                # modify the label size to fit the image
                self.ui.phantom_image_label.setMaximumSize(
                    self.original_phantom_image.shape[1], self.original_phantom_image.shape[0])
                self.ui.phantom_image_label.setMinimumSize(
                    self.original_phantom_image.shape[1], self.original_phantom_image.shape[0])

            if image is None:
                img = QImage(image_path)
            else:
                img = QImage(image.data, image.shape[1], image.shape[0],
                             image.strides[0], QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(img)
            self.ui.phantom_image_label.setPixmap(pixmap)
        except Exception as e:
            print(e)

    def mousePressEvent(self, event):
        try:
            if event.button() == Qt.LeftButton:
                self.dragging = True
                self.prev_pos = event.pos()
        except Exception as e:
            print(e)

    def mouseReleaseEvent(self, event):
        try:
            if event.button() == Qt.LeftButton:
                self.dragging = False
                self.prev_pos = None
        except Exception as e:
            print(e)

    def mouseMoveEvent(self, event):
        try:
            self.scroll_flag = True
            if self.dragging:
                delta = event.pos() - self.prev_pos
                self.prev_pos = event.pos()

                orig_image = self.original_phantom_image.copy().astype(np.int32)

                if ((delta.x() > 0) and (delta.y() == 0)):
                    self.brightness += 10
                elif ((delta.x() < 0) and (delta.y() == 0)):
                    self.brightness -= 10

                self.img_enhanced = orig_image + self.brightness

                # find the values that are greater than 255 and set them to 255
                self.img_enhanced[self.img_enhanced > 255] = 255
                self.img_enhanced[self.img_enhanced < 0] = 0
                self.img_enhanced = self.img_enhanced.astype(np.uint8)

                # get the unique values
                self.most_frequent = np.unique(
                    self.img_enhanced, return_counts=False)

                np.savetxt('./txt_files/most_frequent.txt',
                           np.sort(self.most_frequent), fmt='%d')

                qImage = QImage(self.img_enhanced, self.img_enhanced.shape[1], self.img_enhanced.shape[0],
                                self.img_enhanced.strides[0], QImage.Format_Grayscale8)
                pixmap = QPixmap.fromImage(qImage)
                self.ui.phantom_image_label.setPixmap(pixmap)
        except Exception as e:
            print(e)

    def handle_wheel_event(self, event):
        try:
            self.scroll_flag = True
            # make a copy of  the original image to a new integer numpy array
            orig_image = self.original_phantom_image.copy().astype(np.int32)
            if event.angleDelta().y() > 0:
                self.brightness += 10
            else:
                self.brightness -= 10

            self.img_enhanced = orig_image + self.brightness

            # find the values that are greater than 255 and set them to 255
            self.img_enhanced[self.img_enhanced > 255] = 255
            self.img_enhanced[self.img_enhanced < 0] = 0
            self.img_enhanced = self.img_enhanced.astype(np.uint8)

            # get the unique values
            self.most_frequent = np.unique(
                self.img_enhanced, return_counts=False)

            np.savetxt('./txt_files/most_frequent.txt',
                       np.sort(self.most_frequent), fmt='%d')

            qImage = QImage(self.img_enhanced, self.img_enhanced.shape[1], self.img_enhanced.shape[0],
                            self.img_enhanced.strides[0], QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(qImage)
            self.ui.phantom_image_label.setPixmap(pixmap)
        except Exception as e:
            print(e)

    def handle_mouse_press(self, event):
        try:
            if not self.dragging:
                if self.ui.comboBox.currentText() == 'Show Phantom Image':
                    self.dragging = True
                    self.prev_pos = event.pos()
                    if self.scroll_flag:
                        self.show_image_on_label(
                            None, self.img_enhanced)
                    else:
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
                else:
                    print("Please select the phantom image first")
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
            self.PDWeight = [70, 57, 95, 119, 120, 121, 122, 146, 254, 255]

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
            else:
                self.ui.label_4.setText('0')
                self.ui.label_5.setText('0')
                self.ui.label_6.setText('0')

                if self.ui.comboBox.currentIndex() == 1:
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
            image = self.ui.phantom_image_label.pixmap().toImage()
            # convert the QImage to matrix
            image = qimage2ndarray.rgb_view(image)
            # convert the matrix to grayscale
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            index = self.ui.comboBox_2.currentIndex()
            if index == 0:
                return
            if index == 1:
                # get the image on the label

                self.phantom_image_resized = cv2.resize(
                    image, (16, 16))

                cv2.imwrite('./images/phantom_modified/16x16.png',
                            self.phantom_image_resized)

            elif index == 2:
                self.phantom_image_resized = cv2.resize(
                    image, (32, 32))

                cv2.imwrite('./images/phantom_modified/32x32.png',
                            self.phantom_image_resized)

            elif index == 3:
                self.phantom_image_resized = cv2.resize(
                    image, (64, 64))

                cv2.imwrite('./images/phantom_modified/64x64.png',
                            self.phantom_image_resized)

        except Exception as e:
            print(e)

    def apply_noise(self, value):
        try:
            # Generate the noise
            noise = np.zeros_like(self.original_phantom_image)
            cv2.randn(noise, 0, value)

            # Add the noise to the image
            noisy_image = cv2.add(self.original_phantom_image, noise)

            self.show_image_on_label(None, noisy_image)
        except Exception as e:
            print(e)

    def reset_phantom_to_original(self):
        try:
            self.show_image_on_label(self.image_path)
            self.ui.horizontalSlider.setValue(0)
        except Exception as e:
            print(e)

    # MRI Sequence
    def select_acquisition_system(self):
        try:
            if self.ui.comboBox_3.currentIndex() == 0:
                self.popUpErrorMsg(
                    "Error", "Please select an acquisition system")
                return "Error"
            elif self.ui.comboBox_3.currentIndex() == 1:
                self.create_cartesian_kspace()
            elif self.ui.comboBox_3.currentIndex() == 2:
                self.create_spiral_kspace()
            elif self.ui.comboBox_3.currentIndex() == 3:
                self.create_zig_zag_kspace()
            elif self.ui.comboBox_3.currentIndex() == 4:
                self.create_radial_kspace()
        except Exception as e:
            print(e)

    def create_cartesian_kspace(self):
        try:
            self.kspaceIndicesVisisted = np.array([], dtype=int)
            for i in range(0, self.phantom_image_resized.shape[0]):
                for j in range(0, self.phantom_image_resized.shape[1]):
                    self.kspaceIndicesVisisted = np.append(
                        self.kspaceIndicesVisisted, [i, j])
            self.kspaceIndicesVisisted = self.kspaceIndicesVisisted.reshape(
                -1, 2)

        except Exception as e:
            print(e)

    def create_spiral_kspace(self):
        try:
            self.kspaceIndicesVisisted = np.array([], dtype=int)
            top = 0
            bottom = len(self.phantom_image_resized) - 1
            left = 0
            right = len(self.phantom_image_resized[0]) - 1
            direction = "right"

            while top <= bottom and left <= right:
                if direction == "right":
                    for i in range(left, right+1):

                        self.kspaceIndicesVisisted = np.append(
                            self.kspaceIndicesVisisted, [top, i])
                    top += 1
                    direction = "down"
                elif direction == "down":
                    for i in range(top, bottom+1):

                        self.kspaceIndicesVisisted = np.append(
                            self.kspaceIndicesVisisted, [i, right])
                    right -= 1
                    direction = "left"
                elif direction == "left":
                    for i in range(right, left-1, -1):

                        self.kspaceIndicesVisisted = np.append(
                            self.kspaceIndicesVisisted, [bottom, i])
                    bottom -= 1
                    direction = "up"
                elif direction == "up":
                    for i in range(bottom, top-1, -1):

                        self.kspaceIndicesVisisted = np.append(
                            self.kspaceIndicesVisisted, [i, left])
                    left += 1
                    direction = "right"

            self.kspaceIndicesVisisted = self.kspaceIndicesVisisted.reshape(
                -1, 2)
            # reverse the indices
            self.kspaceIndicesVisisted = self.kspaceIndicesVisisted[::-1]
        except Exception as e:
            print(e)

    def create_zig_zag_kspace(self):
        try:
            self.kspaceIndicesVisisted = np.array([], dtype=int)
            n = len(self.phantom_image_resized)
            going_down = True
            for i in range(n):
                if going_down:
                    for j in range(n):
                        self.kspaceIndicesVisisted = np.append(
                            self.kspaceIndicesVisisted, [i, j])
                    going_down = False
                else:
                    for j in range(n-1, -1, -1):
                        self.kspaceIndicesVisisted = np.append(
                            self.kspaceIndicesVisisted, [i, j])
                    going_down = True

            self.kspaceIndicesVisisted = self.kspaceIndicesVisisted.reshape(
                -1, 2)
        except Exception as e:
            print(e)

    def create_radial_kspace(self):
        try:

            self.kspaceIndicesVisisted = np.array([], dtype=int)
            n = len(self.phantom_image_resized)
            row = col = n // 2
            # main arrows from the center
            # top Right
            topRight = np.array([], dtype=int)
            x = row
            y = col
            while x > 0 and y < n - 1:
                x -= 1
                y += 1
                topRight = np.append(topRight, [x, y])

            # bottom Right
            bottomRight = np.array([], dtype=int)
            x = row
            y = col
            while x < n - 1 and y < n - 1:
                x += 1
                y += 1
                bottomRight = np.append(bottomRight, [x, y])

            # bottom Left
            bottomLeft = np.array([], dtype=int)
            x = row
            y = col
            while x < n - 1 and y > 0:
                x += 1
                y -= 1
                bottomLeft = np.append(bottomLeft, [x, y])

            # top Left
            topLeft = np.array([], dtype=int)
            x = row
            y = col
            while x > 0 and y > 0:
                x -= 1
                y -= 1
                topLeft = np.append(topLeft, [x, y])

            # top
            top = np.array([], dtype=int)
            x = row
            y = col
            while x > 0:
                x -= 1
                top = np.append(top, [x, y])

            # right
            right = np.array([], dtype=int)
            x = row
            y = col
            while y < n - 1:
                y += 1
                right = np.append(right, [x, y])

            # bottom
            bottom = np.array([], dtype=int)
            x = row
            y = col
            while x < n - 1:
                x += 1
                bottom = np.append(bottom, [x, y])

            # left
            left = np.array([], dtype=int)
            x = row
            y = col
            while y > 0:
                y -= 1
                left = np.append(left, [x, y])

            # combine all the arrays
            self.kspaceIndicesVisisted = np.concatenate(
                (topRight, bottomRight, bottomLeft, topLeft, top, right, bottom, left))

            # add the center to the first index of the array
            self.kspaceIndicesVisisted = np.insert(
                self.kspaceIndicesVisisted, 0, [row, col])
            # reshape the array
            self.kspaceIndicesVisisted = self.kspaceIndicesVisisted.reshape(
                -1, 2)
        except Exception as e:
            print(e)

    def read_dial_values_and_calculate_ernst(self):
        try:
            tr_value = int(self.ui.dial.value())
            te_value = int(self.ui.dial_3.value())
            t1_value = int(self.ui.dial_2.value())

            ernst_angle = int(np.arccos(
                math.exp(-tr_value/t1_value)) * (180.0 / math.pi))

            self.ui.label_8.setText(str(tr_value))
            self.ui.label_9.setText(str(t1_value))
            self.ui.label_13.setText(str(ernst_angle))
            self.ui.label_18.setText(str(te_value))
        except Exception as e:
            print(e)

    def apply_rf_pulse(self, image, flip_angle):
        try:

            if self.ui.comboBox_2.currentIndex() == 0:
                self.popUpErrorMsg("Error", 'Please select a phantom size')
                return

            if flip_angle == '':
                self.popUpErrorMsg("Error", 'Please enter a flip angle value')
                return

            flip_angle = int(flip_angle)
            rows, columns = image.shape
            theta = flip_angle * np.pi/180  # angle in radians
            self.gx_phases = np.arange(0, 360, 360 / rows)
            self.gy_phases = np.arange(0, 360, 360 / rows)
            # rotation along y axis
            R = np.array([[np.cos(theta), 0, np.sin(theta)],
                          [0, 1, 0],
                          [-np.sin(theta), 0, np.cos(theta)]])

            # define the vector Mo for the entire image
            Mo = np.zeros((rows, columns, 3))
            Mo[:, :, 2] = image
            # Multiply the vector v by the rotation matrix R to get the flipped vector v_flipped
            return np.round(np.matmul(Mo, R.T), 2)

        except Exception as e:
            print(e)

    def apply_gradient_deprecated(self):
        try:
            image_after_rf_pulse = self.apply_rf_pulse(self.phantom_image_resized,
                                                       self.ui.lineEdit.text())

            rows, columns, _ = image_after_rf_pulse.shape
            image_after_rf_pulse = image_after_rf_pulse.reshape(
                rows * columns, 3)
            # print(image_after_rf_pulse)
            new_image_after_rf_pulse = image_after_rf_pulse.copy()
            k_space_2d = np.zeros((rows, columns), dtype=complex)
            k_space = np.ones((rows, columns))
            phases = np.empty((rows, columns))

            # start the progress bar
            self.ui.progressBar.setValue(0)
            total_steps = len(self.gy_phases) * len(self.gx_phases)

            step_count = 0

            for row_index, gy_phase in enumerate(self.gy_phases):

                phases = gy_phase * \
                    np.arange(rows).reshape(-1, 1) + np.zeros((rows, columns))

                phases_backup = phases.copy()

                for column_index, gx_phase in enumerate(self.gx_phases):

                    phases = phases_backup.copy()
                    phases += gx_phase * np.arange(columns)

                    end_phases = phases.reshape(rows * columns, 1)

                    theta = end_phases * np.pi / 180

                    cos_theta = np.cos(theta)
                    sin_theta = np.sin(theta)

                    R = np.stack([
                        cos_theta, sin_theta, np.zeros_like(theta),
                        -sin_theta, cos_theta, np.zeros_like(theta),
                        np.zeros_like(theta), np.zeros_like(
                            theta), np.ones_like(theta)
                    ], axis=-1).reshape(-1, 3, 3)

                    new_image_after_rf_pulse = np.matmul(
                        R, image_after_rf_pulse.reshape(-1, 3, 1)).reshape(-1, 3)

                    sum_x = np.round(np.sum(new_image_after_rf_pulse[:, 0]))
                    sum_y = np.round(np.sum(new_image_after_rf_pulse[:, 1]))

                    k_space_2d[row_index, column_index] = sum_x + 1j * sum_y
                    k_space[row_index, column_index] = np.sqrt(
                        sum_x**2 + sum_y**2)

                    step_count += 1
                    progress_percent = int(step_count / total_steps * 100)
                    self.ui.progressBar.setValue(progress_percent)
                    self.update_kspace(k_space)
                    self.update_image(k_space_2d)

        except Exception as e:
            print(e)

    def apply_rf_gradient_sequence(self):
        try:
            if self.select_acquisition_system() == "Error":
                return
            image_after_rf_pulse = self.apply_rf_pulse(self.phantom_image_resized,
                                                       self.ui.lineEdit.text())

            kspace_length = len(self.kspaceIndicesVisisted)

            rows, columns, _ = image_after_rf_pulse.shape
            kspaceAngle = 2 * np.pi / rows
            k_space_2d = np.zeros((rows, columns), dtype=complex)
            k_space = np.full((rows, columns), float('inf'))

            self.ui.progressBar.setValue(0)
            total_steps = len(self.gy_phases) * len(self.gx_phases)

            step_count = 0

            for index in range(kspace_length):
                u = self.kspaceIndicesVisisted[index, 0]
                v = self.kspaceIndicesVisisted[index, 1]

                image_after_sequence = image_after_rf_pulse.copy()
                for i in range(rows):
                    for j in range(columns):
                        x_angle = i * kspaceAngle * u
                        y_angle = j * kspaceAngle * v
                        angle = x_angle + y_angle

                        rotation_matrix = np.array([[np.cos(angle), np.sin(angle), 0],
                                                    [-np.sin(angle),
                                                        np.cos(angle), 0],
                                                    [0, 0, 1]])
                        Mo_flipped = image_after_rf_pulse[i, j]

                        Mo_flipped = np.dot(rotation_matrix, Mo_flipped)
                        image_after_sequence[i, j] = Mo_flipped

                        # delay_recovery_matrix = self.create_delay_recovery_matrix(
                        #     0.1, i, j)
                        # image_after_sequence[i, j] = np.dot(
                        #     delay_recovery_matrix, image_after_sequence[i, j])

                sum_x = np.round(np.sum(image_after_sequence[:, :, 0]), 2)
                sum_y = np.round(np.sum(image_after_sequence[:, :, 1]), 2)
                k_space_2d[u, v] = sum_x + 1j * sum_y
                k_space[u, v] = np.sqrt(sum_x**2 + sum_y**2)

                step_count += 1
                progress_percent = int(step_count / total_steps * 100)
                self.ui.progressBar.setValue(progress_percent)

                self.update_kspace(k_space)
                self.update_image(k_space_2d)
        except Exception as e:
            print(e)

    def create_delay_recovery_matrix(self, t, i, j):
        t1_value = self.t1WeightedImage[i, j]
        t2_value = self.t2WeightedImage[i, j]
        return np.array([[np.exp(-t / t2_value), 0, 0],
                         [0, np.exp(-t / t2_value), 0],
                         [0, 0, (1 - np.exp(-t / t1_value))]])

    def update_kspace(self, kspace):
        try:
            self.kspace_figure.clear()
            axes = self.kspace_figure.gca()
            axes.set_xticks([])
            axes.set_yticks([])
            axes.xaxis.tick_top()
            axes.xaxis.set_label_text('Kx')
            axes.yaxis.set_label_text('Ky')
            axes.xaxis.set_label_position('top')

            axes.imshow(kspace, cmap='gray')

            self.kspace_figure.canvas.draw()
            self.kspace_figure.canvas.flush_events()
        except Exception as e:
            print(e)

    def update_image(self, kspace_2d):
        try:
            self.reconstructed_image_figure.clear()
            axes = self.reconstructed_image_figure.gca()
            axes.set_xticks([])
            axes.set_yticks([])

            img = np.fft.ifft2(kspace_2d)
            img = np.real(img).astype(np.uint8)

            axes.imshow(img, cmap='gray')
            self.reconstructed_image_figure.canvas.draw()
            self.reconstructed_image_figure.canvas.flush_events()
        except Exception as e:
            print(e)

    # sequence plotting

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
                start, amp, num, half_sin_wave, elevation=10, step=5)
            self.RFplotter.setData(xAxiesVal, yAxiesVal)

            self.starting_TR_postion = xAxiesVal[50]
            self.startingTR_plotter.setData(
                np.repeat(self.starting_TR_postion, 50), np.linspace(-0.5, 11.5, 50))

            if num > 1:
                self.ending_TR_postion = xAxiesVal[150]

                self.endingTR_plotter.setData(
                    np.repeat(self.ending_TR_postion, 50), np.linspace(10, 11.5, 50))
                self.TR_horizontal_line.setData(np.linspace(
                    self.starting_TR_postion, self.ending_TR_postion, 50), np.repeat(11.3, 50))

                if hasattr(self, 'TR_text'):
                    self.ui.signalPlot.removeItem(self.TR_text)

                center = (self.starting_TR_postion + self.ending_TR_postion)

                self.TR_text = pg.TextItem(
                    text=f"TR= {np.round(center,2)} ms", anchor=(0, 0))

                self.TR_text.setPos(
                    self.starting_TR_postion + center / 5, 11.3)
                self.ui.signalPlot.addItem(self.TR_text)

        except Exception as e:
            print(e)

    def plot_Gss(self, start, amp, num=1, inverted=False):
        try:
            if num > 1 and inverted:
                xAxiesVal, yAxiesVal = self.prebGraphData(
                    start, amp, num, square_wave, elevation=7.5, step=1)
                # delete the last 50 points
                xAxiesVal = xAxiesVal[:-50]
                yAxiesVal = yAxiesVal[:-50]

                # update the last 50 points of y_axis to be multiplied by -1 and add 7.5
                yAxiesVal[-50:] = np.multiply(yAxiesVal[-50:], -1) + 7.5 + 7.5

                # add a zero point to the end of the x and y axies
                xAxiesVal.append(xAxiesVal[-1] + 0.01)
                yAxiesVal.append(7.5)

            else:
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

    def plot_Gfe(self, start, amp, num=1, inverted=False):
        try:
            if num > 1 and inverted:
                xAxiesVal, yAxiesVal = self.prebGraphData(
                    start, amp, num, flat_line, elevation=2.5, step=1)
                # delete the first 50 points
                xAxiesVal = xAxiesVal[:-50]
                yAxiesVal = yAxiesVal[:-50]

                # update the first 50 points of y_axis to be multiplied by -1 and add 2.5
                yAxiesVal[:50] = np.multiply(yAxiesVal[:50], -1) + 2.5 + 2.5

                # add a zero point to the end and start of the x and y axies
                yAxiesVal[0] = 2.5
                xAxiesVal.append(xAxiesVal[-1] + 0.01)
                yAxiesVal.append(2.5)

            else:
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

            self.TE_postion = xAxiesVal[50]
            self.TE_plotter.setData(np.repeat(self.TE_postion, 50),
                                    np.linspace(-0.5, 1, 50))
            self.TE_horizontal_line.setData(np.linspace(self.starting_TR_postion, self.TE_postion, 50),
                                            np.repeat(-0.5, 50))

            if hasattr(self, 'TE_text'):
                self.ui.signalPlot.removeItem(self.TE_text)

            center = (self.TE_postion - self.starting_TR_postion)

            self.TE_text = pg.TextItem(
                text=f"TE= {np.round(center,2)} ms", anchor=(0, 0))

            self.TE_text.setPos(self.starting_TR_postion + 0.2, -0.6)

            self.ui.signalPlot.addItem(self.TE_text)

        except Exception as e:
            print(e)

    def plot_GRE_sequence(self):
        try:
            self.plot_Rf(1.0, 1.0)
            self.plot_Gss(1.0, 1.0, 2, True)
            self.plot_Gpe(2, 1.0)
            self.plot_Gfe(2, 1.0, 2, True)
            self.plot_RO(3, 1.0)
        except Exception as e:
            print(e)

    def plot_chosen_sequence(self):
        try:
            if self.ui.comboBox_4.currentText() == "GRE":
                print("GRE")
                self.plot_GRE_sequence()
            elif self.ui.comboBox_4.currentText() == "SpinEcho":
                self.plot_SE_sequence()
            elif self.ui.comboBox_4.currentText() == "SSFP":
                self.plot_FLASH_sequence()

        except Exception as e:
            print(e)

    # assistive functions:
    def popUpErrorMsg(self, title, body):
        try:
            msgBox = QMessageBox()
            msgBox.setWindowTitle(title)
            msgBox.setText(body)
            msgBox.setIcon(msgBox.Critical)
            msgBox.exec_()
        except Exception as e:
            print(e)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())
