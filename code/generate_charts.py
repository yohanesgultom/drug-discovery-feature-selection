'''
Generate additional charts

@author yohanes.gultom@gmail.com
'''

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os

# check if display available
if os.name == 'posix' and "DISPLAY" not in os.environ:
    matplotlib.use('Agg')

# Feature selection execution time
results = {
    'SVM+RFE': 553.128,
    'SVM+WM': 1375.660,
}

df = pd.DataFrame.from_dict(results, orient='index')
df.plot.barh(fontsize=9, legend=False)
plt.suptitle('Feature selection execution time')
plt.title('Using PubChem BioAssay + DUD-E dataset', fontdict={'fontsize': 9})
plt.xlabel('Time (s)')
plt.ylabel('')
plt.savefig('feature_selection_time_comparison.png')

# Classification time comparison
results = [
    {'Dataset': 'PubChem\nBioAssay\n+ DUD-E', 'SVM': 7.6, 'SVM+RFE': 5.6, 'SVM+WM': 3.3},
    {'Dataset': 'Herbal DB', 'SVM': 14.4, 'SVM+RFE': 10.3, 'SVM+WM': 6.1},
]

df = pd.DataFrame.from_dict(results).set_index('Dataset')
df = df[['SVM', 'SVM+RFE', 'SVM+WM']]
df.plot.barh(fontsize=9)
plt.suptitle('Classification time comparison')
plt.xlabel('Time (s)')
plt.ylabel('')
plt.savefig('classification_time_comparison.png')

plt.show()