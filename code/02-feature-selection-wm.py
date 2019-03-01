'''
Use Wrapper Method (WM) to find an optimal feature set by optimizing an SVM classifier using Genetic Algorithm
on pubchem hiv+decoy dataset. The most optimal feature set is saved in a JSON file.

Result:
Optimal number of features: 249
Average best accuracy: 0.9916777363585875

Execution Time (Core i7 5500U, 8 GB, SSD):
real    22m55.660s
user    67m36.263s
sys     0m16.710s

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
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn_genetic import GeneticSelectionCV

# config
dataset_file = '../dataset/dataset.csv'
# dataset_file = '../dataset/dataset_test.csv' # 0.98633
result_file = '02-feature-selection-wm_result.csv'
feature_mask_file = 'WM_GA_SVM_features_mask.json'
plot_img_feat_file = 'WM_GA_SVM_feat_chart.png'
plot_img_acc_file = 'WM_GA_SVM_acc_chart.png'
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
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# GA feature selection
estimator = SVC(kernel='linear', C=0.9, cache_size=1000, max_iter=1000, verbose=verbosity)
selector = GeneticSelectionCV(estimator, scoring='accuracy', n_population=20, n_generations=100, cv=3, caching=True, verbose=verbosity, n_jobs=-1)
selector = selector.fit(X, y)
count = (selector.support_ == True).sum()
print("Optimal number of features: {}".format(count))
max_tuples = selector.logbook_.select('max')
min_tuples = selector.logbook_.select('min')
avg_tuples = selector.logbook_.select('avg')
scores_max, num_feats = zip(*max_tuples)
scores_min, _ = zip(*min_tuples)
scores_avg, _ = zip(*avg_tuples)
print("Average best accuracy: {}".format(max(scores_max)))

# save result to file
with open(result_file, 'w') as fp:
    fp.write(str(selector.logbook_))
    print('Raw result is saved in {}'.format(result_file))

# Note: with full dataset (unbalanced)
# Optimal number of features: 255
# Average best accuracy: 0.9962926604107819

# save features mask
with open(feature_mask_file, 'w') as f:
    sel_features = np.array(feature_names)[selector.support_]
    json.dump(sel_features.tolist(), f)
    print('Feature mask is saved in {}'.format(feature_mask_file))

# check if display available
if os.name == 'posix' and "DISPLAY" not in os.environ:
    matplotlib.use('Agg')

# plot accuracy per generation
plt.figure()
plt.title('Wrapper Method (GA) Feature Selection')
plt.xlabel('Generation')
plt.ylabel('CV Accuracy')
generations = range(1, len(scores_max) + 1)
line_max, = plt.plot(generations, scores_max, color='b', label='Max')
line_avg, = plt.plot(generations, scores_avg, color='y', label='Avg')
line_min, = plt.plot(generations, scores_min, color='r', label='Min')
plt.legend(handles=[line_max, line_avg, line_min])
plt.savefig(plot_img_acc_file)
print('Accuracy chart is saved in {}'.format(plot_img_acc_file))

# plot accuracy per set of features
plt.figure()
plt.title('Wrapper Method (GA) Feature Selection')
plt.xlabel('Generation')
plt.ylabel('Number of features')
plt.plot(range(1, len(num_feats) + 1), num_feats)
plt.savefig(plot_img_feat_file)
plt.show()
print('Features chart is saved in {}'.format(plot_img_feat_file))
