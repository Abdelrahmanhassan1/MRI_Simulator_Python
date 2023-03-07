import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

class ImageViewer(QMainWindow):
    def __init__(self):
        super().__init__()

        # Create a label to display the image
        self.image_label = QLabel(self)
        self.setCentralWidget(self.image_label)

        # Set the window properties
        self.setWindowTitle("Image Viewer")
        self.setGeometry(100, 100, 800, 600)

        # Connect the mousePressEvent to handle pixel selection
        self.image_label.mousePressEvent = self.handle_mouse_press

    def open_image(self):
        # Open a file dialog to select an image file
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.bmp)")

        # Load the image and display it in the label
        image = QImage(file_path)
        pixmap = QPixmap.fromImage(image)
        self.image_label.setPixmap(pixmap)

    def handle_mouse_press(self, event):
        # Get the position of the mouse click
        x = event.pos().x()
        y = event.pos().y()

        # Get the color of the pixel at the clicked position
        pixmap = self.image_label.pixmap()
        if pixmap is not None:
            pixel_color = pixmap.toImage().pixel(x, y)
            intensity = QColor(pixel_color).getRgb()[0]
            print("Pixel intensity:", intensity)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = ImageViewer()
    viewer.show()
    sys.exit(app.exec_())
