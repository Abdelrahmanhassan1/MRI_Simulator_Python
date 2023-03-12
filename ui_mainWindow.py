# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'd:\My-Github\NEWNEWNEW\MRI_Simulator_Python\mainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1212, 835)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.graphicsView = PlotWidget(self.centralwidget)
        self.graphicsView.setGeometry(QtCore.QRect(140, 160, 351, 261))
        self.graphicsView.setObjectName("graphicsView")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(60, 120, 75, 23))
        self.pushButton.setObjectName("pushButton")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(740, 40, 201, 41))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_7.setFont(font)
        self.label_7.setAlignment(QtCore.Qt.AlignCenter)
        self.label_7.setObjectName("label_7")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(560, 410, 47, 13))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(550, 450, 47, 13))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(540, 510, 47, 13))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(640, 400, 47, 13))
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(610, 470, 47, 13))
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(640, 520, 47, 13))
        self.label_6.setObjectName("label_6")
        self.phantom_image_label = QtWidgets.QLabel(self.centralwidget)
        self.phantom_image_label.setGeometry(QtCore.QRect(740, 250, 281, 211))
        self.phantom_image_label.setText("")
        self.phantom_image_label.setObjectName("phantom_image_label")
        self.comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox.setGeometry(QtCore.QRect(610, 210, 191, 22))
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox_2 = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_2.setGeometry(QtCore.QRect(900, 130, 201, 22))
        self.comboBox_2.setObjectName("comboBox_2")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1212, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "Browse"))
        self.label_7.setText(_translate("MainWindow", "Shepp Logan Phantom"))
        self.label.setText(_translate("MainWindow", "T1"))
        self.label_2.setText(_translate("MainWindow", "T2"))
        self.label_3.setText(_translate("MainWindow", "PD"))
        self.label_4.setText(_translate("MainWindow", "0"))
        self.label_5.setText(_translate("MainWindow", "0"))
        self.label_6.setText(_translate("MainWindow", "0"))
        self.comboBox.setItemText(0, _translate("MainWindow", "image"))
        self.comboBox.setItemText(1, _translate("MainWindow", "t1 "))
        self.comboBox.setItemText(2, _translate("MainWindow", "t2"))
        self.comboBox.setItemText(3, _translate("MainWindow", "t3"))
        self.comboBox_2.setItemText(0, _translate("MainWindow", "change size"))
        self.comboBox_2.setItemText(1, _translate("MainWindow", "size 1"))
        self.comboBox_2.setItemText(2, _translate("MainWindow", "size 2"))
        self.comboBox_2.setItemText(3, _translate("MainWindow", "size 3"))
from pyqtgraph import PlotWidget
