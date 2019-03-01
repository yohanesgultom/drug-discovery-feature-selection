'''
Train 3 classifiers using pubchem hiv+decoy dataset and test it against top 10 herbal Indonesia dataset:
1. SVM with all features
2. SVM with WM-GA feature selection mask (output of 02-feature-selection-wm.py)
3. SVM with SVM-RE feature selection mask (output of 02-feature-selection-svm-rfe.py)

Result:
SVM + RFE Accuracy: 0.6000
SVM + WM Accuracy: 0.7000
SVM Accuracy: 0.5000

Execution Time (Core i7 5500U, 8 GB, SSD):
real    0m40.904s
user    0m33.426s
sys     0m0.845s

@author yohanes.gultom@gmail.com
'''

# suppress sklearn warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pandas
import numpy as np
import json
import os
import csv
from pprint import pprint
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib
import matplotlib.pyplot as plt

# config
train_file = '../dataset/dataset.csv' # pubchem
test_file = '../dataset/dataset_test.csv' # top 10 herbaldb
result_file = '03-evaluate-2_result.csv'
methods = [
    {'name': 'SVM + RFE', 'filename': 'SVM_RFE_features_mask.json'},
    {'name': 'SVM + WM', 'filename': 'WM_GA_SVM_features_mask.json'},
    {'name': 'SVM', 'filename': None},
]
scores_chart_file = '03-evaluate-2_scores_chart.png'
verbosity = 0

# read dataset
df_train = pandas.read_csv(train_file, index_col=0)
df_test = pandas.read_csv(test_file, index_col=0)

results = []
for method in methods:
    name = method['name']
    feature_mask_file = method['filename']    
    if feature_mask_file:
        # apply features mask        
        with open(feature_mask_file, 'r') as f:        
            feature_mask = json.load(f)
        
        # split to data X and labels y
        X_train = df_train[feature_mask].values.astype('float32')
        y_train = df_train['Class'].values
        X_test = df_test[feature_mask].values.astype('float32')
        y_test = df_test['Class'].values
    else:
        # no mask
        X_train = df_train[df_train.columns.drop('Class')].values.astype('float32')
        y_train = df_train['Class'].values
        X_test = df_test[df_test.columns.drop('Class')].values.astype('float32')
        y_test = df_test['Class'].values

    # scale data
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # evaluate
    estimator = SVC(kernel='linear', C=0.9, probability=True, max_iter=1000, verbose=verbosity)
    estimator.fit(X_train, y_train)
    y_pred_proba = estimator.predict_proba(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # accuracy, sensitivity, specificity, precision
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    acc = (tp + tn) * 1.0 / (tp + tn + fp + fn)
    sens = tp * 1.0 / (tp + fn) if tp > 0 else 0
    spec = tn * 1.0 / (tn + fp) if tn > 0 else 0
    prec = tp * 1.0 / (tp + fp) if tp > 0 else 0

    # print result
    features = ', '.join(feature_mask) if feature_mask_file else 'All'    
    results.append({
        'name': name,
        'features': features,
        'num_features': len(feature_mask) if feature_mask_file else X_train.shape[1],
        'acc': acc,
        'sens': sens,
        'spec': spec,
        'prec': prec,
    })

if len(results) > 0:
    # print results
    for i, row in enumerate(results):
        print('{} Accuracy: {:.4f}'.format(row['name'], row['acc']))

    # save result to file
    with open(result_file, 'w') as f:
        w = csv.DictWriter(f, results[0].keys(), delimiter='\t')
        w.writeheader()
        for row in results:        
            w.writerow(row)
        print('Raw result is saved in {}'.format(result_file))

    # check if display available
    if os.name == 'posix' and "DISPLAY" not in os.environ:
        matplotlib.use('Agg')

    # plot acc, sens, spec, prec
    colors = ['blue', 'red', 'orange']
    bar_width = 0.2
    groups = ('Accuracy', 'Sensitivity', 'Precision', 'Specificity')
    index = np.arange(len(groups))
    fig, ax = plt.subplots()    
    for i, row in enumerate(results):
        data = [row['acc'], row['sens'], row['prec'], row['spec']]
        plt.bar(index + (bar_width * i), data, bar_width, color=colors[i], label=row['name'])
    plt.ylabel('Scores')
    plt.ylim([0.0, 1.0])
    plt.suptitle('Performance Comparison on Top 10 Herbal DB')
    plt.title('PubChem BioAssay + DUD-E as training data', fontdict={'fontsize':9})
    plt.xticks(index + bar_width, groups)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=5)
    plt.savefig(scores_chart_file)
    print('Scores chart is saved in {}'.format(scores_chart_file))    

    # show plots
    plt.show()