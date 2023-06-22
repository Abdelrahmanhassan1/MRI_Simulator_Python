# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'd:\My-Github\MRI_Simulator_Python\mainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1804, 635)
        MainWindow.setMinimumSize(QtCore.QSize(1421, 620))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout_22 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_22.setObjectName("horizontalLayout_22")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_19 = QtWidgets.QLabel(self.centralwidget)
        self.label_19.setMinimumSize(QtCore.QSize(0, 0))
        self.label_19.setMaximumSize(QtCore.QSize(16777215, 40))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_19.setFont(font)
        self.label_19.setAlignment(QtCore.Qt.AlignCenter)
        self.label_19.setObjectName("label_19")
        self.verticalLayout.addWidget(self.label_19)
        self.comboBox = QtWidgets.QComboBox(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.comboBox.setFont(font)
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.verticalLayout.addWidget(self.comboBox)
        self.horizontalLayout_15 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_15.setObjectName("horizontalLayout_15")
        self.horizontalSlider = QtWidgets.QSlider(self.centralwidget)
        self.horizontalSlider.setMaximumSize(QtCore.QSize(400, 20))
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setObjectName("horizontalSlider")
        self.horizontalLayout_15.addWidget(self.horizontalSlider)
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setMinimumSize(QtCore.QSize(0, 20))
        self.label_7.setMaximumSize(QtCore.QSize(16777215, 20))
        self.label_7.setAlignment(QtCore.Qt.AlignCenter)
        self.label_7.setObjectName("label_7")
        self.horizontalLayout_15.addWidget(self.label_7)
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setObjectName("pushButton_3")
        self.horizontalLayout_15.addWidget(self.pushButton_3)
        self.verticalLayout.addLayout(self.horizontalLayout_15)
        self.phantom_image_label = QtWidgets.QLabel(self.centralwidget)
        self.phantom_image_label.setMinimumSize(QtCore.QSize(480, 480))
        self.phantom_image_label.setMaximumSize(QtCore.QSize(10000, 10000))
        self.phantom_image_label.setText("")
        self.phantom_image_label.setObjectName("phantom_image_label")
        self.verticalLayout.addWidget(self.phantom_image_label)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setMaximumSize(QtCore.QSize(16777215, 40))
        self.label_2.setStyleSheet("border: 1px solid black;")
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_9.addWidget(self.label_2)
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setMaximumSize(QtCore.QSize(16777215, 40))
        self.label_5.setStyleSheet("border: 1px solid black;")
        self.label_5.setObjectName("label_5")
        self.horizontalLayout_9.addWidget(self.label_5)
        self.horizontalLayout_4.addLayout(self.horizontalLayout_9)
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setMaximumSize(QtCore.QSize(16777215, 40))
        self.label.setStyleSheet("border: 1px solid black;")
        self.label.setObjectName("label")
        self.horizontalLayout_8.addWidget(self.label)
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setMaximumSize(QtCore.QSize(16777215, 40))
        self.label_4.setStyleSheet("border: 1px solid black;")
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_8.addWidget(self.label_4)
        self.horizontalLayout_4.addLayout(self.horizontalLayout_8)
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setMaximumSize(QtCore.QSize(16777215, 40))
        self.label_3.setStyleSheet("border: 1px solid black;")
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_10.addWidget(self.label_3)
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setMaximumSize(QtCore.QSize(16777215, 40))
        self.label_6.setStyleSheet("border: 1px solid black;")
        self.label_6.setObjectName("label_6")
        self.horizontalLayout_10.addWidget(self.label_6)
        self.horizontalLayout_4.addLayout(self.horizontalLayout_10)
        self.verticalLayout.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_7.addLayout(self.verticalLayout)
        self.verticalLayout_24 = QtWidgets.QVBoxLayout()
        self.verticalLayout_24.setObjectName("verticalLayout_24")
        self.verticalLayout_23 = QtWidgets.QVBoxLayout()
        self.verticalLayout_23.setObjectName("verticalLayout_23")
        self.verticalLayout_14 = QtWidgets.QVBoxLayout()
        self.verticalLayout_14.setObjectName("verticalLayout_14")
        self.label_16 = QtWidgets.QLabel(self.centralwidget)
        self.label_16.setMaximumSize(QtCore.QSize(16777215, 40))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_16.setFont(font)
        self.label_16.setAlignment(QtCore.Qt.AlignCenter)
        self.label_16.setObjectName("label_16")
        self.verticalLayout_14.addWidget(self.label_16)
        self.horizontalLayout_13 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_13.setObjectName("horizontalLayout_13")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.browseFileBtn = QtWidgets.QPushButton(self.centralwidget)
        self.browseFileBtn.setMinimumSize(QtCore.QSize(0, 0))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.browseFileBtn.setFont(font)
        self.browseFileBtn.setObjectName("browseFileBtn")
        self.verticalLayout_3.addWidget(self.browseFileBtn)
        self.updateBtn = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.updateBtn.setFont(font)
        self.updateBtn.setObjectName("updateBtn")
        self.verticalLayout_3.addWidget(self.updateBtn)
        self.horizontalLayout_2.addLayout(self.verticalLayout_3)
        self.verticalLayout_7 = QtWidgets.QVBoxLayout()
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.comboBox_5 = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_5.setObjectName("comboBox_5")
        self.comboBox_5.addItem("")
        self.comboBox_5.addItem("")
        self.comboBox_5.addItem("")
        self.comboBox_5.addItem("")
        self.verticalLayout_7.addWidget(self.comboBox_5)
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setObjectName("pushButton_4")
        self.verticalLayout_7.addWidget(self.pushButton_4)
        self.horizontalLayout_2.addLayout(self.verticalLayout_7)
        self.verticalLayout_8 = QtWidgets.QVBoxLayout()
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.comboBox_4 = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_4.setObjectName("comboBox_4")
        self.comboBox_4.addItem("")
        self.comboBox_4.addItem("")
        self.comboBox_4.addItem("")
        self.comboBox_4.addItem("")
        self.verticalLayout_8.addWidget(self.comboBox_4)
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setObjectName("pushButton")
        self.verticalLayout_8.addWidget(self.pushButton)
        self.horizontalLayout_2.addLayout(self.verticalLayout_8)
        self.horizontalLayout_13.addLayout(self.horizontalLayout_2)
        self.verticalLayout_14.addLayout(self.horizontalLayout_13)
        self.verticalLayout_23.addLayout(self.verticalLayout_14)
        self.verticalLayout_22 = QtWidgets.QVBoxLayout()
        self.verticalLayout_22.setObjectName("verticalLayout_22")
        self.horizontalLayout_18 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_18.setObjectName("horizontalLayout_18")
        self.label_25 = QtWidgets.QLabel(self.centralwidget)
        self.label_25.setObjectName("label_25")
        self.horizontalLayout_18.addWidget(self.label_25)
        self.lineEdit_2 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.horizontalLayout_18.addWidget(self.lineEdit_2)
        self.verticalLayout_22.addLayout(self.horizontalLayout_18)
        self.horizontalLayout_19 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_19.setObjectName("horizontalLayout_19")
        self.label_26 = QtWidgets.QLabel(self.centralwidget)
        self.label_26.setObjectName("label_26")
        self.horizontalLayout_19.addWidget(self.label_26)
        self.lineEdit_3 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.horizontalLayout_19.addWidget(self.lineEdit_3)
        self.verticalLayout_22.addLayout(self.horizontalLayout_19)
        self.verticalLayout_15 = QtWidgets.QVBoxLayout()
        self.verticalLayout_15.setObjectName("verticalLayout_15")
        self.horizontalLayout_20 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_20.setObjectName("horizontalLayout_20")
        self.label_27 = QtWidgets.QLabel(self.centralwidget)
        self.label_27.setObjectName("label_27")
        self.horizontalLayout_20.addWidget(self.label_27)
        self.lineEdit_4 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.horizontalLayout_20.addWidget(self.lineEdit_4)
        self.verticalLayout_15.addLayout(self.horizontalLayout_20)
        self.horizontalLayout_21 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_21.setObjectName("horizontalLayout_21")
        self.label_28 = QtWidgets.QLabel(self.centralwidget)
        self.label_28.setObjectName("label_28")
        self.horizontalLayout_21.addWidget(self.label_28)
        self.lineEdit_5 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_5.setObjectName("lineEdit_5")
        self.horizontalLayout_21.addWidget(self.lineEdit_5)
        self.verticalLayout_15.addLayout(self.horizontalLayout_21)
        self.verticalLayout_22.addLayout(self.verticalLayout_15)
        self.verticalLayout_23.addLayout(self.verticalLayout_22)
        self.verticalLayout_24.addLayout(self.verticalLayout_23)
        self.signalPlot = PlotWidget(self.centralwidget)
        self.signalPlot.setMinimumSize(QtCore.QSize(500, 400))
        self.signalPlot.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.signalPlot.setObjectName("signalPlot")
        self.verticalLayout_24.addWidget(self.signalPlot)
        self.horizontalLayout_7.addLayout(self.verticalLayout_24)
        self.verticalLayout_29 = QtWidgets.QVBoxLayout()
        self.verticalLayout_29.setObjectName("verticalLayout_29")
        self.verticalLayout_11 = QtWidgets.QVBoxLayout()
        self.verticalLayout_11.setSizeConstraint(QtWidgets.QLayout.SetMaximumSize)
        self.verticalLayout_11.setObjectName("verticalLayout_11")
        self.dial_3 = QtWidgets.QDial(self.centralwidget)
        self.dial_3.setMinimumSize(QtCore.QSize(0, 60))
        self.dial_3.setMaximumSize(QtCore.QSize(16777215, 40))
        font = QtGui.QFont()
        font.setPointSize(6)
        self.dial_3.setFont(font)
        self.dial_3.setMinimum(400)
        self.dial_3.setMaximum(990)
        self.dial_3.setObjectName("dial_3")
        self.verticalLayout_11.addWidget(self.dial_3)
        self.horizontalLayout_14 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_14.setObjectName("horizontalLayout_14")
        self.label_17 = QtWidgets.QLabel(self.centralwidget)
        self.label_17.setMaximumSize(QtCore.QSize(16777215, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_17.setFont(font)
        self.label_17.setStyleSheet("border: 1px solid black;")
        self.label_17.setAlignment(QtCore.Qt.AlignCenter)
        self.label_17.setObjectName("label_17")
        self.horizontalLayout_14.addWidget(self.label_17)
        self.label_18 = QtWidgets.QLabel(self.centralwidget)
        self.label_18.setMaximumSize(QtCore.QSize(16777215, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_18.setFont(font)
        self.label_18.setStyleSheet("border: 1px solid black;")
        self.label_18.setAlignment(QtCore.Qt.AlignCenter)
        self.label_18.setObjectName("label_18")
        self.horizontalLayout_14.addWidget(self.label_18)
        self.verticalLayout_11.addLayout(self.horizontalLayout_14)
        self.verticalLayout_29.addLayout(self.verticalLayout_11)
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setSizeConstraint(QtWidgets.QLayout.SetMaximumSize)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.dial = QtWidgets.QDial(self.centralwidget)
        self.dial.setMinimumSize(QtCore.QSize(0, 60))
        self.dial.setMaximumSize(QtCore.QSize(16777215, 60))
        font = QtGui.QFont()
        font.setPointSize(6)
        self.dial.setFont(font)
        self.dial.setStyleSheet("border: 2px solid black;")
        self.dial.setMinimum(400)
        self.dial.setMaximum(990)
        self.dial.setObjectName("dial")
        self.verticalLayout_4.addWidget(self.dial)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_10 = QtWidgets.QLabel(self.centralwidget)
        self.label_10.setMaximumSize(QtCore.QSize(16777215, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_10.setFont(font)
        self.label_10.setStyleSheet("border: 1px solid black;")
        self.label_10.setAlignment(QtCore.Qt.AlignCenter)
        self.label_10.setObjectName("label_10")
        self.horizontalLayout.addWidget(self.label_10)
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setMaximumSize(QtCore.QSize(16777215, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_8.setFont(font)
        self.label_8.setStyleSheet("border: 1px solid black;")
        self.label_8.setAlignment(QtCore.Qt.AlignCenter)
        self.label_8.setObjectName("label_8")
        self.horizontalLayout.addWidget(self.label_8)
        self.verticalLayout_4.addLayout(self.horizontalLayout)
        self.verticalLayout_29.addLayout(self.verticalLayout_4)
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setSizeConstraint(QtWidgets.QLayout.SetMaximumSize)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.dial_2 = QtWidgets.QDial(self.centralwidget)
        self.dial_2.setMinimumSize(QtCore.QSize(0, 60))
        self.dial_2.setMaximumSize(QtCore.QSize(16777215, 60))
        font = QtGui.QFont()
        font.setPointSize(6)
        self.dial_2.setFont(font)
        self.dial_2.setMinimum(400)
        self.dial_2.setMaximum(550)
        self.dial_2.setObjectName("dial_2")
        self.verticalLayout_5.addWidget(self.dial_2)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_11 = QtWidgets.QLabel(self.centralwidget)
        self.label_11.setMaximumSize(QtCore.QSize(16777215, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_11.setFont(font)
        self.label_11.setStyleSheet("border: 1px solid black;")
        self.label_11.setAlignment(QtCore.Qt.AlignCenter)
        self.label_11.setObjectName("label_11")
        self.horizontalLayout_3.addWidget(self.label_11)
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_9.setMaximumSize(QtCore.QSize(16777215, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_9.setFont(font)
        self.label_9.setStyleSheet("border: 1px solid black;")
        self.label_9.setAlignment(QtCore.Qt.AlignCenter)
        self.label_9.setObjectName("label_9")
        self.horizontalLayout_3.addWidget(self.label_9)
        self.verticalLayout_5.addLayout(self.horizontalLayout_3)
        self.verticalLayout_29.addLayout(self.verticalLayout_5)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.label_12 = QtWidgets.QLabel(self.centralwidget)
        self.label_12.setMaximumSize(QtCore.QSize(16777215, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_12.setFont(font)
        self.label_12.setStyleSheet("border: 1px solid black;")
        self.label_12.setObjectName("label_12")
        self.horizontalLayout_5.addWidget(self.label_12)
        self.label_13 = QtWidgets.QLabel(self.centralwidget)
        self.label_13.setMaximumSize(QtCore.QSize(16777215, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_13.setFont(font)
        self.label_13.setStyleSheet("border: 1px solid black;")
        self.label_13.setAlignment(QtCore.Qt.AlignCenter)
        self.label_13.setObjectName("label_13")
        self.horizontalLayout_5.addWidget(self.label_13)
        self.verticalLayout_2.addLayout(self.horizontalLayout_5)
        self.verticalLayout_6 = QtWidgets.QVBoxLayout()
        self.verticalLayout_6.setContentsMargins(-1, 12, -1, -1)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.verticalLayout_2.addLayout(self.verticalLayout_6)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.comboBox_2 = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_2.setMaximumSize(QtCore.QSize(16777215, 30))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.comboBox_2.setFont(font)
        self.comboBox_2.setObjectName("comboBox_2")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.horizontalLayout_6.addWidget(self.comboBox_2)
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setMinimumSize(QtCore.QSize(200, 0))
        self.lineEdit.setMaximumSize(QtCore.QSize(400, 30))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.lineEdit.setFont(font)
        self.lineEdit.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.lineEdit.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit.setObjectName("lineEdit")
        self.horizontalLayout_6.addWidget(self.lineEdit)
        self.verticalLayout_2.addLayout(self.horizontalLayout_6)
        self.verticalLayout_26 = QtWidgets.QVBoxLayout()
        self.verticalLayout_26.setContentsMargins(-1, 12, -1, -1)
        self.verticalLayout_26.setObjectName("verticalLayout_26")
        self.verticalLayout_2.addLayout(self.verticalLayout_26)
        self.horizontalLayout_17 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_17.setObjectName("horizontalLayout_17")
        self.label_24 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_24.setFont(font)
        self.label_24.setObjectName("label_24")
        self.horizontalLayout_17.addWidget(self.label_24)
        self.comboBox_7 = QtWidgets.QComboBox(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.comboBox_7.setFont(font)
        self.comboBox_7.setObjectName("comboBox_7")
        self.comboBox_7.addItem("")
        self.comboBox_7.addItem("")
        self.horizontalLayout_17.addWidget(self.comboBox_7)
        self.verticalLayout_2.addLayout(self.horizontalLayout_17)
        self.verticalLayout_27 = QtWidgets.QVBoxLayout()
        self.verticalLayout_27.setContentsMargins(-1, 12, -1, -1)
        self.verticalLayout_27.setObjectName("verticalLayout_27")
        self.verticalLayout_2.addLayout(self.verticalLayout_27)
        self.horizontalLayout_12 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_12.setObjectName("horizontalLayout_12")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setMinimumSize(QtCore.QSize(140, 25))
        self.pushButton_2.setMaximumSize(QtCore.QSize(16777215, 16777215))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setStyleSheet("background-color:orange;\n"
"border-radius:5px\n"
"")
        self.pushButton_2.setObjectName("pushButton_2")
        self.horizontalLayout_12.addWidget(self.pushButton_2)
        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.horizontalLayout_12.addWidget(self.progressBar)
        self.verticalLayout_2.addLayout(self.horizontalLayout_12)
        self.verticalLayout_28 = QtWidgets.QVBoxLayout()
        self.verticalLayout_28.setContentsMargins(-1, 12, -1, -1)
        self.verticalLayout_28.setObjectName("verticalLayout_28")
        self.verticalLayout_2.addLayout(self.verticalLayout_28)
        self.verticalLayout_29.addLayout(self.verticalLayout_2)
        self.horizontalLayout_7.addLayout(self.verticalLayout_29)
        self.verticalLayout_25 = QtWidgets.QVBoxLayout()
        self.verticalLayout_25.setObjectName("verticalLayout_25")
        self.verticalLayout_21 = QtWidgets.QVBoxLayout()
        self.verticalLayout_21.setObjectName("verticalLayout_21")
        self.label_22 = QtWidgets.QLabel(self.centralwidget)
        self.label_22.setMaximumSize(QtCore.QSize(16777215, 20))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_22.setFont(font)
        self.label_22.setAlignment(QtCore.Qt.AlignCenter)
        self.label_22.setObjectName("label_22")
        self.verticalLayout_21.addWidget(self.label_22)
        self.comboBox_3 = QtWidgets.QComboBox(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.comboBox_3.setFont(font)
        self.comboBox_3.setObjectName("comboBox_3")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.verticalLayout_21.addWidget(self.comboBox_3)
        self.horizontalLayout_11 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_11.setObjectName("horizontalLayout_11")
        self.verticalLayout_10 = QtWidgets.QVBoxLayout()
        self.verticalLayout_10.setObjectName("verticalLayout_10")
        self.label_14 = QtWidgets.QLabel(self.centralwidget)
        self.label_14.setMinimumSize(QtCore.QSize(0, 20))
        self.label_14.setMaximumSize(QtCore.QSize(16777215, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_14.setFont(font)
        self.label_14.setAlignment(QtCore.Qt.AlignCenter)
        self.label_14.setObjectName("label_14")
        self.verticalLayout_10.addWidget(self.label_14)
        self.verticalLayout_9 = QtWidgets.QVBoxLayout()
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.verticalLayout_10.addLayout(self.verticalLayout_9)
        self.horizontalLayout_11.addLayout(self.verticalLayout_10)
        self.verticalLayout_13 = QtWidgets.QVBoxLayout()
        self.verticalLayout_13.setObjectName("verticalLayout_13")
        self.label_15 = QtWidgets.QLabel(self.centralwidget)
        self.label_15.setMinimumSize(QtCore.QSize(0, 20))
        self.label_15.setMaximumSize(QtCore.QSize(16777215, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_15.setFont(font)
        self.label_15.setAlignment(QtCore.Qt.AlignCenter)
        self.label_15.setObjectName("label_15")
        self.verticalLayout_13.addWidget(self.label_15)
        self.verticalLayout_12 = QtWidgets.QVBoxLayout()
        self.verticalLayout_12.setObjectName("verticalLayout_12")
        self.verticalLayout_13.addLayout(self.verticalLayout_12)
        self.horizontalLayout_11.addLayout(self.verticalLayout_13)
        self.verticalLayout_21.addLayout(self.horizontalLayout_11)
        self.verticalLayout_25.addLayout(self.verticalLayout_21)
        self.verticalLayout_20 = QtWidgets.QVBoxLayout()
        self.verticalLayout_20.setObjectName("verticalLayout_20")
        self.label_23 = QtWidgets.QLabel(self.centralwidget)
        self.label_23.setMaximumSize(QtCore.QSize(16777215, 20))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_23.setFont(font)
        self.label_23.setAlignment(QtCore.Qt.AlignCenter)
        self.label_23.setObjectName("label_23")
        self.verticalLayout_20.addWidget(self.label_23)
        self.comboBox_6 = QtWidgets.QComboBox(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.comboBox_6.setFont(font)
        self.comboBox_6.setObjectName("comboBox_6")
        self.comboBox_6.addItem("")
        self.comboBox_6.addItem("")
        self.comboBox_6.addItem("")
        self.comboBox_6.addItem("")
        self.verticalLayout_20.addWidget(self.comboBox_6)
        self.horizontalLayout_16 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_16.setObjectName("horizontalLayout_16")
        self.verticalLayout_16 = QtWidgets.QVBoxLayout()
        self.verticalLayout_16.setObjectName("verticalLayout_16")
        self.label_20 = QtWidgets.QLabel(self.centralwidget)
        self.label_20.setMinimumSize(QtCore.QSize(0, 20))
        self.label_20.setMaximumSize(QtCore.QSize(16777215, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_20.setFont(font)
        self.label_20.setAlignment(QtCore.Qt.AlignCenter)
        self.label_20.setObjectName("label_20")
        self.verticalLayout_16.addWidget(self.label_20)
        self.verticalLayout_17 = QtWidgets.QVBoxLayout()
        self.verticalLayout_17.setObjectName("verticalLayout_17")
        self.verticalLayout_16.addLayout(self.verticalLayout_17)
        self.horizontalLayout_16.addLayout(self.verticalLayout_16)
        self.verticalLayout_18 = QtWidgets.QVBoxLayout()
        self.verticalLayout_18.setObjectName("verticalLayout_18")
        self.label_21 = QtWidgets.QLabel(self.centralwidget)
        self.label_21.setMinimumSize(QtCore.QSize(0, 20))
        self.label_21.setMaximumSize(QtCore.QSize(16777215, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_21.setFont(font)
        self.label_21.setAlignment(QtCore.Qt.AlignCenter)
        self.label_21.setObjectName("label_21")
        self.verticalLayout_18.addWidget(self.label_21)
        self.verticalLayout_19 = QtWidgets.QVBoxLayout()
        self.verticalLayout_19.setObjectName("verticalLayout_19")
        self.verticalLayout_18.addLayout(self.verticalLayout_19)
        self.horizontalLayout_16.addLayout(self.verticalLayout_18)
        self.verticalLayout_20.addLayout(self.horizontalLayout_16)
        self.verticalLayout_25.addLayout(self.verticalLayout_20)
        self.horizontalLayout_7.addLayout(self.verticalLayout_25)
        self.horizontalLayout_22.addLayout(self.horizontalLayout_7)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_19.setText(_translate("MainWindow", "Phantom Viewer"))
        self.comboBox.setItemText(0, _translate("MainWindow", "Show Phantom Image"))
        self.comboBox.setItemText(1, _translate("MainWindow", "Show  T1 Weighted Image"))
        self.comboBox.setItemText(2, _translate("MainWindow", "Show T2 Weighted Image"))
        self.comboBox.setItemText(3, _translate("MainWindow", "Show  PD Weighted Image"))
        self.label_7.setText(_translate("MainWindow", "Apply Noise"))
        self.pushButton_3.setText(_translate("MainWindow", "Reset"))
        self.label_2.setText(_translate("MainWindow", "T2:"))
        self.label_5.setText(_translate("MainWindow", "0"))
        self.label.setText(_translate("MainWindow", "T1:"))
        self.label_4.setText(_translate("MainWindow", "0"))
        self.label_3.setText(_translate("MainWindow", "PD:"))
        self.label_6.setText(_translate("MainWindow", "0"))
        self.label_16.setText(_translate("MainWindow", "MRI Sequence"))
        self.browseFileBtn.setText(_translate("MainWindow", "Browse"))
        self.updateBtn.setText(_translate("MainWindow", "Update"))
        self.comboBox_5.setItemText(0, _translate("MainWindow", "Choose Prep Pulse"))
        self.comboBox_5.setItemText(1, _translate("MainWindow", "IR (T1-Prep)"))
        self.comboBox_5.setItemText(2, _translate("MainWindow", "T2-Prep"))
        self.comboBox_5.setItemText(3, _translate("MainWindow", "Tagging Sequence"))
        self.pushButton_4.setText(_translate("MainWindow", "Plot"))
        self.comboBox_4.setItemText(0, _translate("MainWindow", "Choose Sequence"))
        self.comboBox_4.setItemText(1, _translate("MainWindow", "GRE"))
        self.comboBox_4.setItemText(2, _translate("MainWindow", "SpinEcho"))
        self.comboBox_4.setItemText(3, _translate("MainWindow", "SSFP"))
        self.pushButton.setText(_translate("MainWindow", "Plot"))
        self.label_25.setText(_translate("MainWindow", "Inversion delay"))
        self.label_26.setText(_translate("MainWindow", "T2 Prep duration"))
        self.label_27.setText(_translate("MainWindow", "Orientation / Angle"))
        self.label_28.setText(_translate("MainWindow", "Tag Width"))
        self.label_17.setText(_translate("MainWindow", "TE(ms):"))
        self.label_18.setText(_translate("MainWindow", "400"))
        self.label_10.setText(_translate("MainWindow", "TR(ms):"))
        self.label_8.setText(_translate("MainWindow", "400"))
        self.label_11.setText(_translate("MainWindow", "T1 (ms):"))
        self.label_9.setText(_translate("MainWindow", "400"))
        self.label_12.setText(_translate("MainWindow", "Recommended Ernst Angle:"))
        self.label_13.setText(_translate("MainWindow", "90"))
        self.comboBox_2.setItemText(0, _translate("MainWindow", "choose phantom size"))
        self.comboBox_2.setItemText(1, _translate("MainWindow", "16 x16 "))
        self.comboBox_2.setItemText(2, _translate("MainWindow", "32 x 32"))
        self.comboBox_2.setItemText(3, _translate("MainWindow", "64 x 64"))
        self.lineEdit.setPlaceholderText(_translate("MainWindow", "Enter Flip Angle: e.x 90"))
        self.label_24.setText(_translate("MainWindow", "Choose Viewer"))
        self.comboBox_7.setItemText(0, _translate("MainWindow", "Viewer 1"))
        self.comboBox_7.setItemText(1, _translate("MainWindow", "Viewer 2"))
        self.pushButton_2.setText(_translate("MainWindow", "Apply Sequence"))
        self.label_22.setText(_translate("MainWindow", "Viewer 1"))
        self.comboBox_3.setItemText(0, _translate("MainWindow", "Choose Acquisition System"))
        self.comboBox_3.setItemText(1, _translate("MainWindow", "Cartesian"))
        self.comboBox_3.setItemText(2, _translate("MainWindow", "Spiral"))
        self.comboBox_3.setItemText(3, _translate("MainWindow", "Zig Zag"))
        self.label_14.setText(_translate("MainWindow", "K_Space"))
        self.label_15.setText(_translate("MainWindow", "Reconstructed image"))
        self.label_23.setText(_translate("MainWindow", "Viewer 2"))
        self.comboBox_6.setItemText(0, _translate("MainWindow", "Choose Acquisition System"))
        self.comboBox_6.setItemText(1, _translate("MainWindow", "Cartesian"))
        self.comboBox_6.setItemText(2, _translate("MainWindow", "Spiral"))
        self.comboBox_6.setItemText(3, _translate("MainWindow", "Zig Zag"))
        self.label_20.setText(_translate("MainWindow", "K_Space"))
        self.label_21.setText(_translate("MainWindow", "Reconstructed image"))
from pyqtgraph import PlotWidget