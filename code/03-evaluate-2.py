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

import matplotlib.pyplot as plt
from evaluate_performance import evaluate2

evaluate2(
    '../dataset/dataset.csv', # pubchem
    '../dataset/dataset_test.csv',  # top 10 herbaldb
    '03-evaluate-2_result.csv',
    '03-evaluate-2_roc_chart.png',
    'ROC on Herbal DB Dataset top 10',
    'PubChem BioAssay + DUD-E as training data',
    '03-evaluate-2_scores_chart.png',
    'Performance Comparison on Herbal DB top 10',
    'PubChem BioAssay + DUD-E as training data'
)

# show plots
plt.show()
