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
        MainWindow.resize(1471, 725)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(10, 30, 589, 513))
        self.layoutWidget.setObjectName("layoutWidget")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.layoutWidget)
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.comboBox = QtWidgets.QComboBox(self.layoutWidget)
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.verticalLayout_2.addWidget(self.comboBox)
        self.verticalLayout_14 = QtWidgets.QVBoxLayout()
        self.verticalLayout_14.setObjectName("verticalLayout_14")
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.label = QtWidgets.QLabel(self.layoutWidget)
        self.label.setObjectName("label")
        self.horizontalLayout_8.addWidget(self.label)
        self.label_4 = QtWidgets.QLabel(self.layoutWidget)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_8.addWidget(self.label_4)
        self.verticalLayout_14.addLayout(self.horizontalLayout_8)
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.label_2 = QtWidgets.QLabel(self.layoutWidget)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_9.addWidget(self.label_2)
        self.label_5 = QtWidgets.QLabel(self.layoutWidget)
        self.label_5.setObjectName("label_5")
        self.horizontalLayout_9.addWidget(self.label_5)
        self.verticalLayout_14.addLayout(self.horizontalLayout_9)
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.label_3 = QtWidgets.QLabel(self.layoutWidget)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_10.addWidget(self.label_3)
        self.label_6 = QtWidgets.QLabel(self.layoutWidget)
        self.label_6.setObjectName("label_6")
        self.horizontalLayout_10.addWidget(self.label_6)
        self.verticalLayout_14.addLayout(self.horizontalLayout_10)
        self.verticalLayout_2.addLayout(self.verticalLayout_14)
        self.horizontalLayout_4.addLayout(self.verticalLayout_2)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_7 = QtWidgets.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_7.setFont(font)
        self.label_7.setAlignment(QtCore.Qt.AlignCenter)
        self.label_7.setObjectName("label_7")
        self.verticalLayout.addWidget(self.label_7)
        self.phantom_image_label = QtWidgets.QLabel(self.layoutWidget)
        self.phantom_image_label.setMinimumSize(QtCore.QSize(480, 480))
        self.phantom_image_label.setMaximumSize(QtCore.QSize(480, 480))
        self.phantom_image_label.setText("")
        self.phantom_image_label.setObjectName("phantom_image_label")
        self.verticalLayout.addWidget(self.phantom_image_label)
        self.horizontalLayout_4.addLayout(self.verticalLayout)
        self.comboBox_2 = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_2.setGeometry(QtCore.QRect(670, 180, 131, 22))
        self.comboBox_2.setObjectName("comboBox_2")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(710, 220, 75, 23))
        self.pushButton.setObjectName("pushButton")
        self.k_space_label = QtWidgets.QLabel(self.centralwidget)
        self.k_space_label.setGeometry(QtCore.QRect(700, 310, 171, 151))
        self.k_space_label.setText("")
        self.k_space_label.setObjectName("k_space_label")
        self.reconstructed_image_label = QtWidgets.QLabel(self.centralwidget)
        self.reconstructed_image_label.setGeometry(QtCore.QRect(670, 510, 171, 151))
        self.reconstructed_image_label.setText("")
        self.reconstructed_image_label.setObjectName("reconstructed_image_label")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(780, 240, 75, 23))
        self.pushButton_2.setObjectName("pushButton_2")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(440, 600, 160, 80))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout_7.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(920, 20, 402, 439))
        self.widget.setObjectName("widget")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.browseFileBtn = QtWidgets.QPushButton(self.widget)
        self.browseFileBtn.setMinimumSize(QtCore.QSize(80, 29))
        self.browseFileBtn.setObjectName("browseFileBtn")
        self.horizontalLayout_2.addWidget(self.browseFileBtn)
        self.updateBtn = QtWidgets.QPushButton(self.widget)
        self.updateBtn.setObjectName("updateBtn")
        self.horizontalLayout_2.addWidget(self.updateBtn)
        self.verticalLayout_3.addLayout(self.horizontalLayout_2)
        self.signalPlot = PlotWidget(self.widget)
        self.signalPlot.setMinimumSize(QtCore.QSize(400, 400))
        self.signalPlot.setObjectName("signalPlot")
        self.verticalLayout_3.addWidget(self.signalPlot)
        self.widget1 = QtWidgets.QWidget(self.centralwidget)
        self.widget1.setGeometry(QtCore.QRect(620, 20, 214, 148))
        self.widget1.setObjectName("widget1")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.widget1)
        self.verticalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.dial = QtWidgets.QDial(self.widget1)
        font = QtGui.QFont()
        font.setPointSize(6)
        self.dial.setFont(font)
        self.dial.setMinimum(400)
        self.dial.setMaximum(990)
        self.dial.setObjectName("dial")
        self.verticalLayout_4.addWidget(self.dial)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_10 = QtWidgets.QLabel(self.widget1)
        self.label_10.setAlignment(QtCore.Qt.AlignCenter)
        self.label_10.setObjectName("label_10")
        self.horizontalLayout.addWidget(self.label_10)
        self.label_8 = QtWidgets.QLabel(self.widget1)
        self.label_8.setAlignment(QtCore.Qt.AlignCenter)
        self.label_8.setObjectName("label_8")
        self.horizontalLayout.addWidget(self.label_8)
        self.verticalLayout_4.addLayout(self.horizontalLayout)
        self.horizontalLayout_6.addLayout(self.verticalLayout_4)
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.dial_2 = QtWidgets.QDial(self.widget1)
        font = QtGui.QFont()
        font.setPointSize(6)
        self.dial_2.setFont(font)
        self.dial_2.setMinimum(400)
        self.dial_2.setMaximum(550)
        self.dial_2.setObjectName("dial_2")
        self.verticalLayout_5.addWidget(self.dial_2)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_11 = QtWidgets.QLabel(self.widget1)
        self.label_11.setAlignment(QtCore.Qt.AlignCenter)
        self.label_11.setObjectName("label_11")
        self.horizontalLayout_3.addWidget(self.label_11)
        self.label_9 = QtWidgets.QLabel(self.widget1)
        self.label_9.setAlignment(QtCore.Qt.AlignCenter)
        self.label_9.setObjectName("label_9")
        self.horizontalLayout_3.addWidget(self.label_9)
        self.verticalLayout_5.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_6.addLayout(self.verticalLayout_5)
        self.verticalLayout_6.addLayout(self.horizontalLayout_6)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.label_12 = QtWidgets.QLabel(self.widget1)
        self.label_12.setObjectName("label_12")
        self.horizontalLayout_5.addWidget(self.label_12)
        self.label_13 = QtWidgets.QLabel(self.widget1)
        self.label_13.setObjectName("label_13")
        self.horizontalLayout_5.addWidget(self.label_13)
        self.verticalLayout_6.addLayout(self.horizontalLayout_5)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.comboBox.setItemText(0, _translate("MainWindow", "Image"))
        self.comboBox.setItemText(1, _translate("MainWindow", "T1 Weighted"))
        self.comboBox.setItemText(2, _translate("MainWindow", "T2 Weighted"))
        self.comboBox.setItemText(3, _translate("MainWindow", "Proton Density"))
        self.label.setText(_translate("MainWindow", "T1:"))
        self.label_4.setText(_translate("MainWindow", "0"))
        self.label_2.setText(_translate("MainWindow", "T2:"))
        self.label_5.setText(_translate("MainWindow", "0"))
        self.label_3.setText(_translate("MainWindow", "PD:"))
        self.label_6.setText(_translate("MainWindow", "0"))
        self.label_7.setText(_translate("MainWindow", "Shepp Logan Phantom"))
        self.comboBox_2.setItemText(0, _translate("MainWindow", "choose phantom size"))
        self.comboBox_2.setItemText(1, _translate("MainWindow", "16"))
        self.comboBox_2.setItemText(2, _translate("MainWindow", "32"))
        self.comboBox_2.setItemText(3, _translate("MainWindow", "64"))
        self.pushButton.setText(_translate("MainWindow", "PushButton"))
        self.pushButton_2.setText(_translate("MainWindow", "PushButton"))
        self.browseFileBtn.setText(_translate("MainWindow", "Browse"))
        self.updateBtn.setText(_translate("MainWindow", "Update"))
        self.label_10.setText(_translate("MainWindow", "TR(ms):"))
        self.label_8.setText(_translate("MainWindow", "400"))
        self.label_11.setText(_translate("MainWindow", "T1 (ms):"))
        self.label_9.setText(_translate("MainWindow", "400"))
        self.label_12.setText(_translate("MainWindow", "Recommended Ernst Angle:"))
        self.label_13.setText(_translate("MainWindow", "90"))
from pyqtgraph import PlotWidget
