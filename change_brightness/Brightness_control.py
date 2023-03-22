from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QColor
from PyQt5.QtWidgets import QApplication, QLabel


class MyLabel(QLabel):
    def __init__(self, parent=None):
        super(MyLabel, self).__init__(parent)
        self.setMouseTracking(True)
        self.dragging = False
        self.prev_pos = None
        self.brightness_factor = 0.8

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = True
            self.prev_pos = event.pos()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = False
            self.prev_pos = None

    def mouseMoveEvent(self, event):
        if self.dragging:
            delta = event.pos() - self.prev_pos
            print(f"event.pos() = {event.pos().x()} , {event.pos().y()}")
            print(f"self.prev_pos = {self.prev_pos.x()} , {self.prev_pos.y()}")
            print(f"delta = {delta.x()} , {delta.y()}")
            # self.brightness_factor += delta.y() / 100.0
            self.prev_pos = event.pos()
            self.update_image()

    def update_image(self):
        print(f"self.dragging = {self.dragging}")
        image = self.pixmap().toImage()
        width, height = image.width(), image.height()
        for x in range(width):
            for y in range(height):
                color = QColor(image.pixel(x, y))
                h, s, v, a = color.getHsv()

                # print(f"v before = {v}")
                v = min(255, int(v * self.brightness_factor))
                # print(v)
                color.setHsv(h, s, v, a)
                # print(f"v afetr = {v}")
                image.setPixelColor(x, y, color)
        new_pixmap = QPixmap.fromImage(image)
        self.setPixmap(new_pixmap)


# Create a label and set a pixmap
app = QApplication([])
label = MyLabel()
pixmap = QPixmap("Shepp_logan.png")
label.setPixmap(pixmap)

# Show the label
label.show()
app.exec_()
#
# import sys
# from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QSizePolicy
# from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor
# from PyQt5.QtCore import Qt
# import cv2
#
# class ImageBrightnessApp(QMainWindow):
#     def __init__(self, image_path):
#         super().__init__()
#
#         self.image = cv2.imread(image_path)
#         self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
#
#         self.display_image = QImage(
#             self.image.data, self.image.shape[1], self.image.shape[0], QImage.Format_RGB888
#         )
#
#         self.label = QLabel(self)
#         self.label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
#         self.label.setPixmap(QPixmap.fromImage(self.display_image))
#
#         self.setCentralWidget(self.label)
#
#         self.setWindowTitle('Image Brightness')
#         self.setGeometry(100, 100, self.image.shape[1], self.image.shape[0])
#
#     def change_brightness(self, value):
#         brightness_img = cv2.addWeighted(self.image, 1, self.image * 0, value, 0)
#         self.display_image = QImage(
#             brightness_img.data, brightness_img.shape[1], brightness_img.shape[0], QImage.Format_RGB888
#         )
#         self.label.setPixmap(QPixmap.fromImage(self.display_image))
#
#     def mouseMoveEvent(self, event):
#         value = (event.y() - (self.image.shape[0] / 2)) / (self.image.shape[0] / 2)
#         self.change_brightness(value)
#
# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#
#     image_path = './Shepp_logan.png'
#
#     mainWin = ImageBrightnessApp(image_path)
#     mainWin.show()
#
#     sys.exit(app.exec_())
#
