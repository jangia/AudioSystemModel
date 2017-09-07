import datetime
import pymongo
from pymongo import MongoClient
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
DATA_RANGE = 26

client = MongoClient()
db = client.amp
fft_all = pd.DataFrame(list(db.fft_amp_phi.find({'gain': '4', 'volume': '4', 'frequency': '440.0'})))
fft_all['amp'] = np.float64(fft_all['amp'])
sorted_fft = fft_all.sort_values(by='amp', ascending=True)
# Initialize X
#X_amp = np.array([fft_all['fft_amp'][i] for i in range(0, DATA_RANGE)])

X_ph = np.array([sorted_fft['fft_ph'][i] for i in range(0, DATA_RANGE)])
amps = sorted(sorted_fft['amp'])

lines = []
for i in range(len(X_ph)):
    if  8 < i < 13:#== 0:
        color = '#{0:06X}'.format(i * 600000)
        line = plt.plot(np.unwrap(X_ph[i]), color=color, label='|A| {:.3}'.format(amps[i]))
        lines.append(line[0])
        print(amps[i])

#labels = [line.get_label() for line in lines]
plt.legend(handles=lines, ncol=2, bbox_to_anchor=(1.05, 1), loc=2)

plt.title('Faza posnetka v odvisnosti od frekvence pri sinusnemu vhodnemu signalu s frekvenco 440 Hz')
plt.ylabel('Kot (°)')
plt.xlabel('Frekvenca (Hz)')
plt.grid()
plt.xlim(xmin=0)
plt.savefig('plots/faza_podrobno_{0}.jpg'.format(datetime.datetime.now()), bbox_inches='tight')
plt.show()
