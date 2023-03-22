from collections import Counter
import json
import math
import time
from PyQt5.QtWidgets import QMessageBox
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QPixmap, QImage, QColor, QPainter, QPen, QBrush, QIntValidator
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

        self.image_path = './images/phantom_modified/300px-Shepp_logan.png'
        self.ui.comboBox_2.currentIndexChanged.connect(
            self.change_phantom_size)
        self.ui.comboBox.currentIndexChanged.connect(
            self.handle_image_features_combo_box)
        self.modify_the_image_intensities_distribution(self.image_path)
        self.show_image_on_label(self.image_path)
        self.ui.phantom_image_label.mousePressEvent = self.handle_mouse_press
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

        self.ui.signalPlot.plotItem.hideAxis('left')

        self.RFplotter = self.ui.signalPlot.plot([], [], pen=self.redPen)
        self.GSSplotter = self.ui.signalPlot.plot([], [], pen=self.greenPen)
        self.GPEplotter = self.ui.signalPlot.plot([], [], pen=self.bluePen)
        self.GFEplotter = self.ui.signalPlot.plot([], [], pen=self.redPen)
        self.ROplotter = self.ui.signalPlot.plot([], [], pen=self.greenPen)
        self.startingTR_plotter = self.ui.signalPlot.plot(
            [], [], pen=self.whitePen)
        self.TE_plotter = self.ui.signalPlot.plot([], [], pen=self.whitePen)
        self.TE_horizontal_title = self.ui.signalPlot.plot(
            [], [], pen=self.whitePen)

        self.ui.browseFileBtn.released.connect(self.browseFile)
        self.ui.updateBtn.released.connect(self.update)

        # RF and Gradient
        self.ui.pushButton.released.connect(
            lambda: self.apply_rf_pulse(self.phantom_image_resized, self.ui.lineEdit.text()))
        self.ui.pushButton_2.released.connect(
            lambda: self.apply_gradient(self.new_3D_matrix_image))

        self.read_dial_values_and_calculate_ernst()
        for dial in [self.ui.dial, self.ui.dial_2, self.ui.dial_3]:
            dial.valueChanged.connect(
                self.read_dial_values_and_calculate_ernst)

    @QtCore.pyqtSlot()
    def show_image_on_label(self, image_path, image=None):
        try:
            self.original_phantom_image = cv2.imread(image_path, 0)
            # modify the label size to fit the image
            self.ui.phantom_image_label.setMaximumSize(
                self.original_phantom_image.shape[1], self.original_phantom_image.shape[0])
            self.ui.phantom_image_label.setMinimumSize(
                self.original_phantom_image.shape[1], self.original_phantom_image.shape[0])

            self.mean, self.std_dev = cv2.meanStdDev(
                self.original_phantom_image)
            if image is None:
                img = QImage(image_path)
            else:
                img = QImage(image, image.shape[1], image.shape[0],
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
            if self.dragging:
                delta = event.pos() - self.prev_pos
                # print(f"event.pos() = {event.pos().x()} , {event.pos().y()}")
                # print(f"self.prev_pos = {self.prev_pos.x()} , {self.prev_pos.y()}")
                # print(f"delta = {delta.x()} , {delta.y()}")
                # self.brightness_factor += delta.y() / 100.0
                self.prev_pos = event.pos()
                if ((delta.x() > 0) and (delta.y() == 0) and (self.brightness_factor < 250)):
                    self.brightness_factor += 2
                    # print(f"self.brightness_factor = {self.brightness_factor}")
                    self.update_brightness()
                elif ((delta.x() < 0) and (delta.y() == 0) and (self.brightness_factor > -250)):
                    self.brightness_factor += -2
                    # print(f"self.brightness_factor = {self.brightness_factor}")
                    self.update_brightness()
        except Exception as e:
            print(e)

    def update_brightness(self):
        try:
            # print(f"self.dragging = {self.dragging}")

            value_manipulated = cv2.add(self.value, self.brightness_factor)
            # print(f"value = {value_manipulated}")
            self.img_hsv_manipulated[:, :, 2] = value_manipulated
            img_rgb = cv2.cvtColor(self.img_hsv_manipulated, cv2.COLOR_HSV2RGB)
            qimage = QImage(
                img_rgb.data, img_rgb.shape[1], img_rgb.shape[0], QImage.Format_RGB888)

            new_pixmap = QPixmap.fromImage(qimage)

            self.ui.phantom_image_label.setPixmap(new_pixmap)
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
            if self.ui.comboBox.currentText() == 'Show Phantom Image':
                if self.scroll_flag:
                    self.show_image_on_label(
                        self.image_path, self.img_enhanced)
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

        except Exception as e:
            print(e)

    # MRI Sequence

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

        except Exception as e:
            print(e)

    def apply_gradient(self, image_after_rf_pulse):
        try:
            if image_after_rf_pulse is None:
                self.popUpErrorMsg("Error", 'Please apply RF pulse first')
                return
            backup_image = image_after_rf_pulse.copy()
            rows, columns, _ = image_after_rf_pulse.shape
            k_space_2d = np.zeros((rows, columns), dtype=complex)
            k_space = np.zeros((rows, columns))
            phases = np.zeros((rows, columns))

            # start the progress bar
            self.ui.progressBar.setValue(0)
            total_steps = len(self.gy_phases) * len(self.gx_phases)

            step_count = 0

            for row_index, gy_phase in enumerate(self.gy_phases):
                for column_index, gx_phase in enumerate(self.gx_phases):
                    image_after_rf_pulse = backup_image.copy()
                    # Update progress bar
                    step_count += 1
                    progress_percent = int(step_count / total_steps * 100)
                    self.ui.progressBar.setValue(progress_percent)
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
                    sum = np.round(
                        np.sum(image_after_rf_pulse, axis=(0, 1)), 2)
                    k_space_2d[row_index][column_index] = np.round(
                        sum[0], 2) - 1j * np.round(sum[1], 2)

                    k_space[row_index, column_index] = np.round(np.sqrt(
                        sum[0]**2 + sum[1]**2), 2)

                    self.update_kspace(k_space)
                    self.update_image(k_space_2d)

        except Exception as e:
            print(e)

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
                start, amp, num, half_sin_wave, elevation=10, oscillation=True)
            self.RFplotter.setData(xAxiesVal, yAxiesVal)

            self.starting_TR_postion = xAxiesVal[50]
            self.startingTR_plotter.setData(
                np.repeat(self.starting_TR_postion, 50), np.linspace(0, 12.5, 50))
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

            self.TE_postion = xAxiesVal[50]
            self.TE_plotter.setData(np.repeat(self.TE_postion, 50),
                                    np.linspace(0, 12.5, 50))
            self.TE_horizontal_title.setData(np.linspace(self.starting_TR_postion, self.TE_postion, 50),
                                             np.repeat(1.8, 50))
            text = pg.TextItem(
                text=f"TE= {np.round(self.TE_postion - self.starting_TR_postion,2)} s", anchor=(0, 0))
            text.setPos(self.starting_TR_postion, 1.8)
            self.ui.signalPlot.addItem(text)
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
