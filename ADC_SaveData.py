import SMG_ADS1256PyLib
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
import time
import pandas as pd
import os


def plot_result(data, dataLength):
    
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(data)

    N= dataLength
    T = 1/(1/0.000240)
    yf = scipy.fftpack.fft(data)
    xf = np.linspace(0.0, 1.0 / (2.0 * T),N/2)

    plt.subplot(2,1,2)
    plt.plot(xf,2.0/N*np.abs(yf[:N//2]))
    plt.xlabel('frequecny')
    plt.ylabel('Amplifier')
    plt.show()

buffer = False
PGA = 0x00
samplingRate = 0xF0
sampleCount = 2000
InitialSPI = SMG_ADS1256PyLib.initialSPI() #Initial RPI SPI I/O pin
SetParameter = SMG_ADS1256PyLib.setADC1256BaseParameter(buffer,PGA,samplingRate) #Args(Buffer, PGA, SamplingRate)
data = SMG_ADS1256PyLib.readDiffChannelVolts(sampleCount) #read AIN0/AIN1 differential channel Args(Sample_count)
data = np.array(data) / 1670000
plot_result(data,sampleCount)
save_data(data)
