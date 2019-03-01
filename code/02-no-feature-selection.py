'''
Train and test SVM on pubchem hiv+decoy dataset without feature selection

Result:
Mean 0.9927670076383615

Execution Time (Core i7 5500U, 8 GB, SSD):
real    0m10.212s
user    0m27.026s
sys     0m0.739s

@author yohanes.gultom@gmail.com
'''

# suppress sklearn warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pandas
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# config
dataset_file = '../dataset/dataset.csv'
# dataset_file = '../dataset/dataset_test.csv' # 0.9807
verbosity = 0

# read dataset
df = pandas.read_csv(dataset_file, index_col=0)
# get columns with nonzero variance
df = df.loc[:, df.var() > 0]

# split to data X and labels y
X = df[df.columns.drop('Class')].values.astype('float32')
y = df['Class'].values

# scale data
# Note: 
# for RBF StandardScaler produce 4% better accuracy than MinMaxScaler
# for Linear MinMaxScaler produce 3% better accuracy than StandardScaler
# scaler = StandardScaler()
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# train SVM 
# Note: already reach 0.991 (RBF) and 0.990 (Linear)
# Using all (unbalanced) data: 0.9967692289459841
# Using 7331 (balanced) data: 0.9893595241285864
svc = SVC(kernel='linear', C=0.9, cache_size=1000, max_iter=1000, verbose=verbosity)
scores = cross_val_score(svc, X, y, cv=10, n_jobs=-1, verbose=verbosity)
print('Mean {}'.format(np.mean(scores)))

