import SMG_ADS1256PyLib
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
import time
import pandas as pd
import os
import json
from keras.models import Sequential, load_model

def model_instance():
    print("start loading model...")
    model = load_model('saved_models/11042019-112212-e1.h5')
    print("loaded model finsihed")
    return model

def save_data(data):
    data = pd.DataFrame(data)
    data.to_csv('Test.csv')

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

def get_test_data(data, seq_len, normalise):
    '''
    Create x, y test data windows
    Warning: batch method, not generative, make sure you have enough memory to
    load data, otherwise reduce size of the training split.
    '''
   
    data_windows = []
    for i in range(len(data) - seq_len):
        data_windows.append(data[i:i+seq_len])

    data_windows = np.array(data_windows).astype(float)
    data_windows = normalise_windows(data_windows, single_window=False) if normalise else data_windows
    data_windows = data_windows.reshape(i+1,seq_len,-1)
    x = data_windows[:, :-1]
    y = data_windows[:, -1, [0]]
    return x,y

def predict_point_by_point(model,data):
    #Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
    print('[Model] Predicting Point-by-Point...')
    predicted = model.predict(data)
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted

def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    model = model_instance()
    configs = json.load(open('config.json', 'r')) 
    buffer = False
    PGA = 0x00
    samplingRate = 0xF0
    sampleCount = 500
    InitialSPI = SMG_ADS1256PyLib.initialSPI() #Initial RPI SPI I/O pin
    SetParameter = SMG_ADS1256PyLib.setADC1256BaseParameter(buffer,PGA,samplingRate) #Args(Buffer, PGA, SamplingRate)
    while(1):
        start_time = time.time()
        data = SMG_ADS1256PyLib.readDiffChannelVolts(sampleCount) #read AIN0/AIN1 differential channel Args(Sample_count)
        data = np.array(data) / 1670000
        x_test, y_test  = get_test_data(data,seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise'])  
        predicted = predict_point_by_point(model,x_test)
        end_time = time.time()
        EST = end_time - start_time
        print('spent time : ',EST)
    endflag = SMG_ADS1256PyLib.endSPIfunc()
    #plot_result(data,sampleCount)
    #save_data(data)
     
    #print('start inference\n')

    plot_results(predicted, y_test)
