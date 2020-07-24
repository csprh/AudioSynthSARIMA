from scipy.io import wavfile
fs, data = wavfile.read('string_1.wav')
import pmdarima as pm
import numpy as np
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

data3 = pm.datasets.load_wineind()
train, test = data3[:150], data[150:]
lenSeq = 1000
subSamp = 40
f0SamplesSS = 10
f0Samples = 400

data.shape
data = data[:lenSeq]
dataSS = data[::subSamp]
#from pyramid.arima import auto_arima
f0Samples = 10 # fs = 44100, f0 = 110 (A2), therefore f0 in samples is approx 400
#thissa = pm.auto_arima(train, error_action='ignore', seasonal=True, m=12)
thissarimaSS =  pm.auto_arima(dataSS, start_p=1, start_q=1,test='adf',  max_p= 3, max_q= 3, max_d = 3,    m=f0SamplesSS,start_P=0,max_D=2,max_Q=2, max_P=2, trace=True,error_action='ignore', suppress_warnings=True)
paramsSS = thissarimaSS.get_params([0])
sos = params.get('seasonal_order')
sosL = list(sos)
sosL[3] = f0Samples
sosT = tuple(sosL)
thissarima.set_params(seasonal_order=(sosT))
yhat = thissarima.predict(n_periods=200)
thisfig = plt.figure(figsize=(12,8))
plt.plot(np.arange(1,lenSeq+1), data, label='Real Sequence', color='blue')
plt.plot(np.arange(lenSeq,lenSeq+len(yhat)), yhat, label='Forecast-',	color='green')
plt.show()
thisfig.savefig("Pred3.pdf", bbox_inches='tight')
plt.close(); print('\n')




thissarima2 = 1
#stepwise_model = auto_arima(data, start_p=1, start_q=1,max_p=1, max_q=1, m=f0Samples ,start_P=0,seasonal=True,d=1, D=1, trace=True, error_action='ignore',                           suppress_warnings=False, stepwise=True)
