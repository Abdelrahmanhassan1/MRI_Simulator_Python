import sys
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QSlider, QVBoxLayout

import numpy as np
import cv2


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        # Load the image
        self.image = cv2.imread("image.jpg")

        # Create the widgets
        self.image_label = QLabel()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)

        # Create the layout
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.slider)
        self.setLayout(layout)

        # Connect the slider signal to the noise function
        self.slider.valueChanged.connect(self.apply_noise)

        # Set the initial image
        self.update_image(self.image)

    def update_image(self, image):
        # Convert the image to a QPixmap and set it on the label
        pixmap = QPixmap.fromImage(
            QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888))
        self.image_label.setPixmap(pixmap)

    def apply_noise(self, value):
        # Convert the image to grayscale
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Generate the noise
        noise = np.zeros_like(gray)
        cv2.randn(noise, 0, value)

        # Add the noise to the image
        noisy_image = cv2.add(gray, noise)

        # Convert the image back to color
        color_image = cv2.cvtColor(noisy_image, cv2.COLOR_GRAY2BGR)

        # Update the image label
        self.update_image(color_image)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
