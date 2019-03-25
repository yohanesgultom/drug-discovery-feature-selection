'''
Train and test 3 classifiers using pubchem hiv+decoy dataset:
1. SVM with all features
2. SVM with WM-GA feature selection mask (output of 02-feature-selection-wm.py)
3. SVM with SVM-RE feature selection mask (output of 02-feature-selection-svm-rfe.py)

Result:
SVM + RFE Accuracy: 0.9902 in 5.6 s
SVM + WM Accuracy: 0.9891 in 3.3 s
SVM Accuracy: 0.9902 in 7.6 s

Execution Time (Core i7 5500U, 8 GB, SSD):
real    1m9.144s
user    0m50.504s
sys     0m1.331s

@author yohanes.gultom@gmail.com
'''

import matplotlib.pyplot as plt
from evaluate_performance import evaluate

evaluate(
    '../dataset/dataset.csv',
    '03-evaluate-1_result.csv',
    '03-evaluate-1_roc_chart.png',
    'ROC on ROC on PubChem BioAssay + DUD-E',
    'As both train and test data',
    '03-evaluate-1_scores_chart.png',
    'Performance Comparison on ROC on PubChem BioAssay + DUD-E',
    'As training and testing data'
)

# show plots
plt.show()
