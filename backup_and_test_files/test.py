from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QApplication, QPushButton, QVBoxLayout, QWidget


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        # Create a QPushButton widget
        self.button = QPushButton('Run', self)
        self.button.setFixedSize(100, 30)

        # Set the initial background color of the button to the desired color
        self.button.setStyleSheet('background-color: red')

        # Connect the button's clicked signal to a custom function that will perform the desired operation
        self.button.clicked.connect(self.run_operation)

        # Create a QVBoxLayout to hold the button
        layout = QVBoxLayout()
        layout.addWidget(self.button)
        self.setLayout(layout)

        self.show()

    def run_operation(self):
        # Set the button's background color to the initial color
        self.button.setStyleSheet('background-color: red')

        # Create a QTimer object to update the button's background color at a regular interval
        timer = QTimer()
        timer.timeout.connect(self.update_button_background_color)
        timer.start(100)

        # Perform the desired operation
        # Update the button's background color to show the progress of the operation
        # Once the operation is complete, disconnect the clicked signal from the custom function and reset the button's background color to its initial value.
        for i in range(100):
            # Perform the operation
            self.button.repaint()
            # Update the button's background color to show the progress of the operation
            self.button.setStyleSheet(
                f'background-color: rgb({255 - i * 2}, {i * 2}, 0)')
        timer.stop()
        self.button.setStyleSheet('background-color: red')
        self.button.clicked.disconnect(self.run_operation)

    def update_button_background_color(self):
        # Update the button's background color
        current_color = self.button.palette().color(self.button.backgroundRole())
        new_color = QColor(current_color.red(), current_color.green(
        ), current_color.blue(), current_color.alpha() + 5)
        self.button.setStyleSheet(f'background-color: {new_color.name()}')


if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    app.exec_()
