'''
Use SVM Recursive Feature Elimination (SVM-RFE) to find an optimal feature set
on pubchem hiv+decoy dataset. The most optimal feature set is saved in a JSON file.

Result:
Optimal number of features: 471
Average accuracy: 0.991541124519848

Execution Time (Core i7 5500U, 8 GB, SSD):
real    9m13.128s
user    22m35.254s
sys     2m35.800s

@author yohanes.gultom@gmail.com
'''


# suppress sklearn warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pandas
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import json
import os
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import RFECV

# config
dataset_file = '../dataset/dataset.csv'
# dataset_file = '../dataset/dataset_test.csv' # 0.9861
result_file = '02-feature-selection-svm-rfe_result.json'
feature_mask_file = 'SVM_RFE_features_mask.json'
plot_img_file = 'SVM_RFE_chart.png'
verbosity = 0

# read dataset
df = pandas.read_csv(dataset_file, index_col=0)
# get columns with nonzero variance
df = df.loc[:, df.var() > 0]
feature_names = list(df[df.columns.drop('Class')])

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

# SVM-RE feature selection with cross-validation
# http://scikit-learn.org/stable/auto_examples/feature_selection/plot_rfe_with_cross_validation.html
svc = SVC(kernel='linear', C=0.9, cache_size=1000, max_iter=1000, verbose=verbosity)
selector = RFECV(estimator=svc, step=1, cv=3, scoring='accuracy', verbose=verbosity, n_jobs=-1)
selector.fit(X, y)
print('Optimal number of features: {}'.format(selector.n_features_))
print('Average accuracy: {}'.format(max(selector.grid_scores_)))
# Note: using full (unbalanced) dataset
# Optimal number of features: 383
# Average accuracy: 0.9966104289695464

# Note: using 7331 data (balanced) dataset
# Optimal number of features: 384
# Average accuracy: 0.9892229313626694

# selected feature names
sel_features = np.array(feature_names)[selector.support_]

# save result
with open(result_file, 'w') as f:
    json.dump({
        'features_by_rank_asc': sorted(zip(sel_features.tolist(), selector.ranking_.tolist()), key=lambda x: x[1]),
        'scores_by_num_features_asc': selector.grid_scores_.tolist(),
    }, f)
    print('Raw result is saved in {}'.format(result_file))

# save features mask
with open(feature_mask_file, 'w') as f:   
    json.dump(sel_features.tolist(), f)
    print('Feature mask is saved in {}'.format(feature_mask_file))

# check if display available
if os.name == 'posix' and "DISPLAY" not in os.environ:
    matplotlib.use('Agg')

# plot
scores = selector.grid_scores_
plt.figure()
plt.title('SVM-RFE Feature Selection')
plt.xlabel('Number of features selected')
plt.ylabel('CV Accuracy')
plt.plot(range(1, len(scores) + 1), scores)
plt.savefig(plot_img_file)
plt.show()
print('Accuracy chart is saved in {}'.format(plot_img_file))