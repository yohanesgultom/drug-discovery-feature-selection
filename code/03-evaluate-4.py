'''
Train and test 3 classifiers using expanded herbal Indonesia dataset (manual docking result):
1. SVM with all features
2. SVM with WM-GA feature selection mask (output of 02-feature-selection-wm.py)
3. SVM with SVM-RE feature selection mask (output of 02-feature-selection-svm-rfe.py)

Result:
SVM + RFE Accuracy: 0.9727 in 0.0 s
SVM + WM Accuracy: 0.9727 in 0.0 s
SVM Accuracy: 0.9700 in 0.0 s

Execution Time (Core i7 5500U, 8 GB, SSD):
real    0m4.454s
user    0m2.374s
sys     0m1.228s

@author yohanes.gultom@gmail.com
'''

# suppress sklearn warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pandas
import numpy as np
from scipy import interp
import json
import os
import csv
from pprint import pprint
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib
import matplotlib.pyplot as plt
import time

# config
dataset_file = '../dataset/dataset_test_expanded.csv' # pubchem
result_file = '03-evaluate-4_result.csv'
methods = [
    {'name': 'SVM + RFE', 'filename': 'SVM_RFE_features_mask.json'},
    {'name': 'SVM + WM', 'filename': 'WM_GA_SVM_features_mask.json'},
    {'name': 'SVM', 'filename': None},
]
roc_chart_file = '03-evaluate-4_roc_chart.png'
scores_chart_file = '03-evaluate-4_scores_chart.png'
verbosity = 0

# read dataset
df = pandas.read_csv(dataset_file, index_col=0)

results = []
for method in methods:
    name = method['name']
    feature_mask_file = method['filename']    
    if feature_mask_file:
        # apply features mask        
        with open(feature_mask_file, 'r') as f:        
            feature_mask = json.load(f)
        
        # split data and label then apply mask        
        X = df[feature_mask].values.astype('float32')
        y = df['Class'].values
    else:
        # no mask
        X = df[df.columns.drop('Class')].values.astype('float32')
        y = df['Class'].values

    # scale data
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # evaluate
    estimator = SVC(kernel='linear', C=0.9, probability=True, max_iter=1000, verbose=verbosity)
    cv = StratifiedKFold(n_splits=3)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    accs = []
    sens = []
    specs = []
    precs = []
    exec_times = []
    for train, test in cv.split(X, y):
        start = time.time()
        probas_ = estimator.fit(X[train], y[train]).predict_proba(X[test])        
        # roc
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        # accuracy, sensitivity, specificity, precision
        y_pred = np.argmax(probas_, axis=1)
        tn, fp, fn, tp = confusion_matrix(y[test], y_pred).ravel()
        accs.append((tp + tn) * 1.0 / (tp + tn + fp + fn))
        sens.append(tp * 1.0 / (tp + fn))
        specs.append(tn * 1.0 / (tn + fp))
        precs.append(tp * 1.0 / (tp + fp))
        exec_times.append(time.time() - start)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    features = ', '.join(feature_mask) if feature_mask_file else 'All'    
    results.append({
        'name': name,
        'features': features,
        'num_features': len(feature_mask) if feature_mask_file else X.shape[1],
        'mean_tpr': mean_tpr,
        'mean_fpr': mean_fpr,
        'mean_auc': mean_auc,
        'std_auc': std_auc,
        'mean_acc': np.mean(accs),
        'mean_sens': np.mean(sens),
        'mean_spec': np.mean(specs),
        'mean_prec': np.mean(precs),
        'mean_time': np.mean(exec_times),
    })

if len(results) > 0:
    # print results
    for i, row in enumerate(results):
        print('{} Accuracy: {:.4f} in {:.1f} s'.format(row['name'], row['mean_acc'], row['mean_time']))

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

    # plot ROC
    colors = ['orange', 'blue', 'red']
    plt.figure()
    handles = []
    for i, row in enumerate(results):
        h, = plt.plot(row['mean_fpr'], row['mean_tpr'], color=colors[i], label=r'Mean ROC {} (AUC = {:.4f} $\pm$ {:.4f})'.format(row['name'], row['mean_auc'], row['std_auc']), lw=2, alpha=.8)
        handles.append(h)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.suptitle('ROC on Herbal DB Dataset')
    plt.title('As both train and test data', fontdict={'fontsize': 9})
    plt.legend(loc='lower right')
    plt.savefig(roc_chart_file)
    print('ROC chart is saved in {}'.format(roc_chart_file))    

    # plot acc, sens, spec, prec
    colors = ['blue', 'red', 'orange']
    bar_width = 0.2
    groups = ('Accuracy', 'Sensitivity', 'Precision', 'Specificity')
    index = np.arange(len(groups))
    fig, ax = plt.subplots()    
    for i, row in enumerate(results):
        data = [row['mean_acc'], row['mean_sens'], row['mean_prec'], row['mean_spec']]
        plt.bar(index + (bar_width * i), data, bar_width, color=colors[i], label=row['name'])
    # plt.yticks(np.arange(0.950, 1.000, 0.005))
    plt.ylim([0.4, 1.0])
    plt.ylabel('Scores')
    plt.suptitle('Performance Comparison on Herbal DB manual docking')
    plt.title('As training and testing data', fontdict={'fontsize': 9})
    plt.xticks(index + bar_width, groups)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=5)
    plt.savefig(scores_chart_file)
    print('Scores chart is saved in {}'.format(scores_chart_file))    

    # show plots
    plt.show()