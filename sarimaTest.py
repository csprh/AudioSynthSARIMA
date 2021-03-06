from scipy.io import wavfile
fs, data = wavfile.read('string_1.wav')
import pmdarima as pm
from pmdarima import pipeline
from pmdarima import model_selection
from pmdarima import preprocessing as ppc
from pmdarima import arima
from pmdarima.arima import StepwiseContext
import numpy as np
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

data3 = pm.datasets.load_wineind()
train, test = data3[:150], data[150:]
lenSeq = 10000
subSamp = 40
f0SamplesSS = 10
f0Samples = 400

data.shape
data = data[:lenSeq]
dataSS = data[::subSamp]

with StepwiseContext(max_steps=2):
  pipe = pipeline.Pipeline([
      ("fourier", ppc.FourierFeaturizer(m=f0Samples)),
      ("arima", arima.AutoARIMA(stepwise=True, maxiter=20, with_intercept = False, start_p=5, start_q=4,  max_p= 6, max_q= 6,  trace=1, error_action="ignore",
                              seasonal=False,  # because we use Fourier
                              suppress_warnings=True))
  ])

  pipe.fit(data)
  yhat = pipe.predict(n_periods=1000)

#from pyramid.arima import auto_arima
#f0Samples = 10 # fs = 44100, f0 = 110 (A2), therefore f0 in samples is approx 400
#thissa = pm.auto_arima(train, error_action='ignore', seasonal=True, m=12)
#thissarimaSS =  pm.auto_arima(dataSS, with_intercept = False, d = 0, D = 0, start_p=0, start_q=0,test='adf',  max_p= 3, max_q= 3,     m=f0SamplesSS,start_P=0, start_Q= 0, max_Q=3, max_P=3, trace=True,error_action='ignore', suppress_warnings=True)

#thissarimaS1 =  pm.auto_arima(data, with_intercept = False, d = 0, D = 0, start_p=0, start_q=0,test='adf',  max_p= 3, max_q= 3,     m=f0Samples,start_P=0, start_Q= 0, max_Q=3, max_P=3, trace=True,error_action='ignore', suppress_warnings=True)
#thissarima =  pm.auto_arima(data, with_intercept = False, d = 0, start_p=0, start_q=0,test='adf',  max_p= 5, max_q= 5,  seasonal=False, trace=True,error_action='ignore', suppress_warnings=True)
#paramsSS = thissarimaSS.get_params([0])
#sos = paramsSS.get('seasonal_order')
#sosL = list(sos)
#sosL[3] = f0Samples
#sosT = tuple(sosL)
#thissarimaSS.set_params(seasonal_order=(sosT))
#yhat = thissarimaSS.fit_predict(data[:-1000], n_periods=2000)
#yhat2 = thissarimaSS.predict(n_periods=200)
thisfig = plt.figure(figsize=(12,8))
plt.plot(np.arange(1,lenSeq+1), data, label='Real Sequence', color='blue')
plt.plot(np.arange(lenSeq,lenSeq+len(yhat)), yhat, label='Forecast-',	color='green')
plt.show()
thisfig.savefig("Pred4.pdf", bbox_inches='tight')
plt.close(); print('\n')




thissarima2 = 1
#stepwise_model = auto_arima(data, start_p=1, start_q=1,max_p=1, max_q=1, m=f0Samples ,start_P=0,seasonal=True,d=1, D=1, trace=True, error_action='ignore',                           suppress_warnings=False, stepwise=True)
