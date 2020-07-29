import configuration as config

from scipy.io import wavfile

import pmdarima as pm
from pmdarima import pipeline
from pmdarima import model_selection
from pmdarima import preprocessing as ppc
from pmdarima import arima
from pmdarima.arima import StepwiseContext
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

fs, data = wavfile.read('string_1.wav')

data = data[:config.lenSeq]

with StepwiseContext(max_steps=config.max_steps):
  pipe = pipeline.Pipeline([
      ("fourier", ppc.FourierFeaturizer(m=config.f0Samples)),
      ("arima", arima.AutoARIMA(stepwise=True, maxiter=config.maxiter, with_intercept = False, start_p=5, start_q=4,  max_p= 6, max_q= 6,  trace=1, error_action="ignore",
                              seasonal=False,  # because we use Fourier
                              suppress_warnings=True))
  ])

  pipe.fit(data)
  yhat = pipe.predict(n_periods=config.n_periods)

thisfig = plt.figure(figsize=(12,8))
plt.plot(np.arange(1,lenSeq+1), data, label='Real Sequence', color='blue')
plt.plot(np.arange(lenSeq,lenSeq+len(yhat)), yhat, label='Forecast-',	color='green')
plt.show()
thisfig.savefig("Pred4.pdf", bbox_inches='tight')
plt.close(); print('\n')

thissarima2 = 1
