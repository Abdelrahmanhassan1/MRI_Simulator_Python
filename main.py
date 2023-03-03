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
        self.show_image()
        self.load_matrices()
        self.ui.image_frame.mousePressEvent = self.get_pixel_intensity
        

    @QtCore.pyqtSlot()
    def show_image(self):
        self.image = cv2.imread('./images/Shepp_logan.png')

        # self.image = cv2.imread('./T1Matrix.jpg')
        # self.image = cv2.rotate(self.image, cv2.ROTATE_90_CLOCKWISE)

        height, width, channel = self.image.shape
        self.heightttt = height
        bytesPerLine = 3 * width
        self.image = QtGui.QImage(self.image.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888).rgbSwapped()
        self.ui.image_frame.setPixmap(QtGui.QPixmap.fromImage(self.image))
        self.ui.image_frame.setScaledContents(True)
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
        
        # write each matrix in a txt file
        np.savetxt('T1Matrix.txt', self.T1Matrix, fmt='%d')
        np.savetxt('T2Matrix.txt', self.T2Matrix, fmt='%d')
        np.savetxt('PDMatrix.txt', self.PDMatrix, fmt='%d')
        self.create_the_corresponding_images()

    def create_the_corresponding_images(self):
        cv2.imwrite('T1Matrix.jpg', self.T1Matrix)
        cv2.imwrite('T2Matrix.jpg', self.T2Matrix)
        cv2.imwrite('PDMatrix.jpg', self.PDMatrix)

    def load_matrices(self):
        self.T1Matrix = np.loadtxt('T1Matrix.txt')
        self.T2Matrix = np.loadtxt('T2Matrix.txt')
        self.PDMatrix = np.loadtxt('PDMatrix.txt')
    
    def get_pixel_intensity(self, event):
        x = event.pos().x()
        y = event.pos().y()
        scaling_factor = self.heightttt / self.ui.image_frame.height()
        x_scaled = int(x * scaling_factor)
        y_scaled = int(y * scaling_factor)
        color = QtGui.QColor(self.image.pixel(x_scaled, y_scaled))
        intensity = color.red()
        print(f"Intensity: {intensity}")
    
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())