import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel
from PyQt5.QtGui import QPixmap, QImage, QColor
from PyQt5.QtCore import Qt
from PyQt5 import QtCore, QtGui, QtWidgets
from mainWindow import Ui_MainWindow

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        # Set the window properties
        self.setWindowTitle("Image Viewer")
        
        self.open_image()
        self.set_image_matrices()
        # Connect the mousePressEvent to handle pixel selection
        self.ui.phantom_image_label.mousePressEvent = self.handle_mouse_press

    def set_image_matrices(self):
        pixmap = self.ui.phantom_image_label.pixmap()
         # Create two matrices with the same shape as the image
        width = pixmap.width()
        height = pixmap.height()
        T1Matrix = [[0] * width for _ in range(height)]
        T2Matrix = [[0] * width for _ in range(height)]
        PDMatrix = [[0] * width for _ in range(height)]

        # Assign unique values to each pixel in the matrices based on the intensity of the corresponding pixel in the image
        # image = pixmap.toImage()
        # for y in range(height):
        #     for x in range(width):
        #         pixel_color = image.pixel(x, y)
        #         pixel_intensity = QColor(pixel_color).getRgb()[0]
        #         # if pixel_intensity == 0:
        #         #     T1Matrix[y][x] = 0
        #         #     T2Matrix[y][x] = 255
        #         #     PDMatrix[y][x] = 255
        #         # elif pixel_intensity == 101:
        #         #     T1Matrix[y][x] = 50
        #         #     T2Matrix[y][x] = 200
        #         #     PDMatrix[y][x] = 255
        #         # elif pixel_intensity == 76:
        #         #     T1Matrix[y][x] = 100
        #         #     T2Matrix[y][x] = 150
        #         #     PDMatrix[y][x] = 255
        #         # elif pixel_intensity == 51:
        #         #     T1Matrix[y][x] = 150
        #         #     T2Matrix[y][x] = 100
        #         #     PDMatrix[y][x] = 255
        #         # elif pixel_intensity == 25:
        #         #     T1Matrix[y][x] = 200
        #         #     T2Matrix[y][x] = 50
        #         #     PDMatrix[y][x] = 255
        #         # elif pixel_intensity == 255:
        #         #     T1Matrix[y][x] = 255
        #         #     T2Matrix[y][x] = 0
        #         #     PDMatrix[y][x] = 255
                    

        # # cv2.imshow("image", PDMatrix)

    def get_all_image_intensities_with_no_duplicates(self):
        # Get the unique intensities in the image
        image = self.ui.phantom_image_label.pixmap().toImage()
        intensities = set()
        for y in range(image.height()):
            for x in range(image.width()):
                pixel_color = image.pixel(x, y)
                intensity = QColor(pixel_color).getRgb()[0]
                intensities.add(intensity)

        print("Unique intensities:", intensities)

    def open_image(self):
        # Open a file dialog to select an image file
        file_path = "./images/256px-Shepp_logan.png"
        # Load the image and display it in the label
        image = QImage(file_path)
        pixmap = QPixmap.fromImage(image)
        self.ui.phantom_image_label.setPixmap(pixmap)
        self.setCentralWidget(self.ui.phantom_image_label)
        # self.ui.phantom_image_label.resize(500, 500)
        # self.ui.phantom_image_label.scaleContents = True
        # self.get_all_image_intensities_with_no_duplicates()
        
    def handle_mouse_press(self, event):
        # Get the position of the mouse click
        x = event.pos().x()
        y = event.pos().y()

        # Get the color of the pixel at the clicked position
        pixmap = self.ui.phantom_image_label.pixmap()
        if pixmap is not None:
            pixel_color = pixmap.toImage().pixel(x, y)
            intensity = QColor(pixel_color).getRgb()[0]
            print("Pixel intensity:", intensity)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())