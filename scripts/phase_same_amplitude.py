import datetime
import pymongo
from pymongo import MongoClient
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
DATA_RANGE = 55

client = MongoClient()
db = client.amp
fft_all = pd.DataFrame(list(db.fft_amp_phi.find({'gain': '4', 'volume': '4', 'amp': '0.5904900000000001'})))
fft_all['frequency'] = np.float64(fft_all['frequency'])
sorted_fft = fft_all.sort_values(by='frequency', ascending=True)
# Initialize X
#X_amp = np.array([fft_all['fft_amp'][i] for i in range(0, DATA_RANGE)])

X_ph = np.array([sorted_fft['fft_ph'][i] for i in range(0, DATA_RANGE)])
amps = sorted(sorted_fft['frequency'])

lines = []
for i in range(len(X_ph)):
    color = '#{0:06X}'.format(i * 300000)
    line = plt.plot(np.unwrap(X_ph[i]), color=color, label='f {:.5}'.format(amps[i]))
    lines.append(line[0])
    print(amps[i])

#labels = [line.get_label() for line in lines]
plt.legend(handles=lines, ncol=2, bbox_to_anchor=(1.05, 1), loc=2)

plt.title('Faza posnetka v odvisnosti od frekvence pri sinusnemu vhodnemu signalu z normirano amplitudo 0.5904900000000001')
plt.ylabel('Kot (°)')
plt.xlabel('Frekvenca (Hz)')
plt.grid()
plt.xlim(xmin=0)
plt.savefig('plots/amplituda_{0}.jpg'.format(datetime.datetime.now()), bbox_inches='tight')
plt.close()
