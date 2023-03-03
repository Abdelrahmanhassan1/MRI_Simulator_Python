from PyQt5 import QtGui, QtCore, QtWidgets
import cv2
import sys
from mainWindow import Ui_MainWindow

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.show_image()
        self.ui.image_frame.mousePressEvent = self.get_pixel_intensity


    @QtCore.pyqtSlot()
    def show_image(self):
        self.image = cv2.imread('./images/Shepp_logan.png')
        height, width, channel = self.image.shape
        self.heightttt = height
        bytesPerLine = 3 * width
        self.image = QtGui.QImage(self.image.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888).rgbSwapped()
        self.ui.image_frame.setPixmap(QtGui.QPixmap.fromImage(self.image))
        self.ui.image_frame.setScaledContents(True)

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