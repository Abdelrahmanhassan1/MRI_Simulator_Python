# from PyQt5 import QtGui, QtCore, QtWidgets
# import cv2
# import sys
# from mainWindow import Ui_MainWindow

# class MainWindow(QtWidgets.QMainWindow):
#     def __init__(self, *args, **kwargs):
#         super(MainWindow, self).__init__()
#         self.ui = Ui_MainWindow()
#         self.ui.setupUi(self)

#         self.show_image()
#         self.ui.image_frame.mousePressEvent = self.get_pixel_intensity


#     @QtCore.pyqtSlot()
#     def show_image(self):
#         self.image = cv2.imread('./images/Shepp_logan.png')
#         height, width, channel = self.image.shape
#         self.heightttt = height
#         bytesPerLine = 3 * width
#         self.image = QtGui.QImage(self.image.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888).rgbSwapped()
#         self.ui.image_frame.setPixmap(QtGui.QPixmap.fromImage(self.image))
#         self.ui.image_frame.setScaledContents(True)

#     def get_pixel_intensity(self, event):
#         x = event.pos().x()
#         y = event.pos().y()
#         scaling_factor = self.heightttt / self.ui.image_frame.height()
#         x_scaled = int(x * scaling_factor)
#         y_scaled = int(y * scaling_factor)
#         color = QtGui.QColor(self.image.pixel(x_scaled, y_scaled))
#         intensity = color.red()
#         print(f"Intensity: {intensity}")
    
# if __name__ == '__main__':
#     app = QtWidgets.QApplication(sys.argv)
#     main = MainWindow()
#     main.show()
#     sys.exit(app.exec_())









from PyQt5 import QtGui, QtCore, QtWidgets
import cv2
import sys
import numpy as np
from mainWindow import Ui_MainWindow


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.comboBox.currentIndexChanged.connect(self.handle_combo_box)

        self.show_image()
        self.load_matrices_from_images()
        self.ui.image_frame.mousePressEvent = self.get_pixel_intensity

    @QtCore.pyqtSlot()
    def show_image(self):
        self.image = cv2.imread('./images/256px-Shepp_logan.png')
        height, width, channel = self.image.shape
        self.heightttt = height
        bytesPerLine = 3 * width
        self.image = QtGui.QImage(self.image.data, width, height,
                                  bytesPerLine, QtGui.QImage.Format_RGB888).rgbSwapped()
        self.ui.image_frame.setPixmap(QtGui.QPixmap.fromImage(self.image))
        self.ui.image_frame.setScaledContents(True)
        # self.setCentralWidget(self.ui.image_frame)
        # self.create_the_corresponding_matrices(height, width)

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

    def get_pixel_intensity(self, event):
        x = event.pos().x()
        y = event.pos().y()
        scaling_factor = self.heightttt / self.ui.image_frame.height()
        x_scaled = int(x * scaling_factor)
        y_scaled = int(y * scaling_factor)
        color = QtGui.QColor(self.image.pixel(x_scaled, y_scaled))
        intensity = color.red()
        # get the values from the matrices
        t1 = self.T1Matrix[x_scaled, y_scaled]
        t2 = self.T2Matrix[x_scaled, y_scaled]
        pd = self.PDMatrix[x_scaled, y_scaled]

        self.ui.label_4.setText(str(t1))
        self.ui.label_5.setText(str(t2))
        self.ui.label_6.setText(str(pd))
        print(f"T1: {t1}")
        print(f"T2: {t2}")
        print(f"PD: {pd}")
        print(f"Intensity: {intensity}")

    def handle_combo_box(self):
        if self.ui.comboBox.currentIndex() == 0:
            self.image = cv2.imread('./images/Shepp_logan.png')
            height, width, channel = self.image.shape
            self.heightttt = height
            bytesPerLine = 3 * width
            self.image = QtGui.QImage(
                self.image.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888).rgbSwapped()
            self.ui.image_frame.setPixmap(QtGui.QPixmap.fromImage(self.image))
            self.ui.image_frame.setScaledContents(True)
        elif self.ui.comboBox.currentIndex() == 1:
            self.T1Matrix = cv2.imread('./images/T1Matrix.jpg')
            height, width, channel = self.T1Matrix.shape
            self.heightttt = height
            bytesPerLine = 3 * width
            self.T1Matrix = QtGui.QImage(
                self.T1Matrix.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888).rgbSwapped()
            self.ui.image_frame.setPixmap(
                QtGui.QPixmap.fromImage(self.T1Matrix))
            self.ui.image_frame.setScaledContents(True)

        elif self.ui.comboBox.currentIndex() == 2:
            self.T2Matrix = cv2.imread('./images/T2Matrix.jpg')
            height, width, channel = self.T2Matrix.shape
            self.heightttt = height
            bytesPerLine = 3 * width
            self.T2Matrix = QtGui.QImage(
                self.T2Matrix.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888).rgbSwapped()
            self.ui.image_frame.setPixmap(
                QtGui.QPixmap.fromImage(self.T2Matrix))
            self.ui.image_frame.setScaledContents(True)
        elif self.ui.comboBox.currentIndex() == 3:
            self.PDMatrix = cv2.imread('./images/PDMatrix.jpg')
            height, width, channel = self.PDMatrix.shape
            self.heightttt = height
            bytesPerLine = 3 * width
            self.PDMatrix = QtGui.QImage(
                self.PDMatrix.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888).rgbSwapped()
            self.ui.image_frame.setPixmap(
                QtGui.QPixmap.fromImage(self.PDMatrix))
            self.ui.image_frame.setScaledContents(True)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())
