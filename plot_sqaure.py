import matplotlib.pyplot as plt
import numpy as np
t = np.arange(start=0,stop= 5,step= 1/100)
############################################
#function create square wave
#input : amplitude of wave , number of point for wave
#output : array with value of wave
############################################
def square_wave(Amp , NumOfPoints):
    return np.full(NumOfPoints, Amp) ##create np array with specific length and value

############################################
#function to set place of square wave before plot
#input : start point of function , current wave that want to set place for it
#output : array ready to plot in specific place
#💡💡if signal will plot in zero we should make start point in 1  to make wave take
############################################
def set_square_wave_place(start ,value_for_sqr_wave):
    value_for_sqr_wave = np.insert(value_for_sqr_wave,0,np.zeros(start)) ## add zeros at first to set start of wanted signal
    return  np.insert(value_for_sqr_wave,len(value_for_sqr_wave),np.zeros(len(t) - NumOfPoints -start))


NumOfPoints = 50
pulse = square_wave(2, NumOfPoints)
plt.plot(t, set_square_wave_place(1,pulse))
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()