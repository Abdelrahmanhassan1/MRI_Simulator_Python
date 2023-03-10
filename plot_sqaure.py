import sys
import pyqtgraph as pg
from ui_mainWindow import Ui_MainWindow
from PyQt5 import QtCore, QtGui, QtWidgets
import matplotlib.pyplot as plt
import numpy as np
t = np.arange(start=0, stop=10, step=1/100)
print(f"length of x axis {len(t)}")
############################################
# function create square wave
# input : amplitude of wave , number of point for wave
# output : array with value of wave
############################################


def square_wave(Amp, NumOfPoints=100):
    return np.full(NumOfPoints, Amp)


def half_sin_wave(Amp, Freq):
    t_sin = np.arange(0, 1, 1/100)
    return Amp * np.sin(np.pi * Freq * t_sin)


def sin_wave(Amp, Freq):
    t_sin = np.arange(0, 1, 1/200)
    return Amp * np.sin(2 * np.pi * Freq * t_sin)


############################################
# function to set place of square wave before plot
# input : start point of function , current wave that want to set place for it
# output : array ready to plot in specific place
# 💡💡if signal will plot in zero we should make start point in 1  to make wave take
############################################
def set_square_wave_place(start, value_for_sqr_wave):

    # add zeros at first to set start of wanted signal
    value_for_sqr_wave = np.insert(value_for_sqr_wave, 0, np.zeros(start))
    value_for_sqr_wave = np.insert(value_for_sqr_wave, len(
        value_for_sqr_wave), np.zeros(len(t) - len(value_for_sqr_wave)))
    return value_for_sqr_wave


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.redPen = pg.mkPen(color=(255, 0, 0), width=2)
        self.greenPen = pg.mkPen(color=(0, 255, 0), width=2)
        self.bluePen = pg.mkPen(color=(0, 0, 255), width=2)
        self.whitePen = pg.mkPen(color=(255, 255, 255))

        self.ui.signalPlot.setXRange(-0.5, 10.5, padding=0)
        self.ui.signalPlot.setYRange(-1, 11.5, padding=0)

        self.ui.signalPlot.plotItem.addLine(y=0, pen=self.whitePen)
        self.ui.signalPlot.plotItem.addLine(y=2.5, pen=self.whitePen)
        self.ui.signalPlot.plotItem.addLine(y=5, pen=self.whitePen)
        self.ui.signalPlot.plotItem.addLine(y=7.5, pen=self.whitePen)
        self.ui.signalPlot.plotItem.addLine(y=10, pen=self.whitePen)

        self.ui.signalPlot.plotItem.setLabel('left', "Amplitude", units='V')
        self.ui.signalPlot.plotItem.setLabel('bottom', "Time", units='s')

        self.RFplotter = self.ui.signalPlot.plot([], [], pen=self.redPen)
        self.GSSplotter = self.ui.signalPlot.plot([], [], pen=self.greenPen)
        self.GPEplotter = self.ui.signalPlot.plot([], [], pen=self.bluePen)
        self.GFEplotter = self.ui.signalPlot.plot([], [], pen=self.redPen)
        self.ROplotter = self.ui.signalPlot.plot([], [], pen=self.greenPen)

        # self.plotter_2.setData(t,2 + set_square_wave_place(1,square_wave(1, 100)))

        # plo thorizontal line in plot at y = 2 with white pen
        self.plot_Rf(1, 1, 3)
        self.plot_Gss(1, 1, 3)
        self.plot_Gpe(3, 1, 2)
        self.plot_Gfe(4, 1, 2)
        self.plot_RO(4, 1)

        # self.plotter_3.setData(t, set_square_wave_place(300,square_wave(1,100))

    def plot_Rf(self, start, amp, num=1):
        yAxiesVal = []
        xAxiesVal = []

        for i in range(num):
            yAxiesVal.extend(10 + (half_sin_wave(amp, 1) * np.power(-1, i)))
            xAxiesVal.extend(np.arange(start, start + 1, 1/100))
            start += 1

        self.RFplotter.setData(xAxiesVal, yAxiesVal)

    def plot_Gss(self, start, amp, num=1):
        yAxiesVal = []
        xAxiesVal = []

        for i in range(num):
            yAxiesVal.extend(7.5 + square_wave(amp))
            yAxiesVal[i*100], yAxiesVal[-1] = 7.5, 7.5
            xAxiesVal.extend(np.arange(start, start + 1, 1/100))
            start += 1

        self.GSSplotter.setData(xAxiesVal, yAxiesVal)

    def plot_Gpe(self, start, amp, num=1):

        yAxiesVal = []
        xAxiesVal = []
        for j in range(num):
            for i in range(5):
                yAxiesVal.extend(5 + square_wave(amp) - i*0.5)
                yAxiesVal[i*100 + 500*j], yAxiesVal[i*100 + 99 + 500*j] = 5, 5
                xAxiesVal.extend(np.arange(start, start + 1, 1/100))

            start += 2

        self.GPEplotter.setData(xAxiesVal, yAxiesVal)

    def plot_Gfe(self, start, amp, num=1):
        yAxiesVal = []
        xAxiesVal = []

        for i in range(num):
            yAxiesVal.extend(2.5 + square_wave(amp))
            yAxiesVal[i*100], yAxiesVal[-1] = 2.5, 2.5
            xAxiesVal.extend(np.arange(start, start + 1, 1/100))
            start += 1

        self.GFEplotter.setData(xAxiesVal, yAxiesVal)

    def plot_RO(self, start, amp, num=1):
        yAxiesVal = 0 + half_sin_wave(amp, 1)
        xAxiesVal = np.arange(start, start + 1, 1/100)

        self.ROplotter.setData(xAxiesVal, yAxiesVal)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())


NumOfPoints = 50
pulse = square_wave(2, NumOfPoints)
sin_wave = sin_wave(3, 1)
plt.plot(t, set_square_wave_place(200, sin_wave))
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()