'''
Train 3 classifiers using pubchem hiv+decoy dataset and test it against expanded herbal Indonesia dataset (manual docking result):
1. SVM with all features
2. SVM with WM-GA feature selection mask (output of 02-feature-selection-wm.py)
3. SVM with SVM-RE feature selection mask (output of 02-feature-selection-svm-rfe.py)

Result:
SVM + RFE Accuracy: 0.6076 in 10.5 s
SVM + WM Accuracy: 0.6894 in 6.2 s
SVM Accuracy: 0.5204 in 14.5 s

Execution Time (Core i7 5500U, 8 GB, SSD):
real    0m37.318s
user    0m34.466s
sys     0m0.738s

@author yohanes.gultom@gmail.com
'''
import matplotlib.pyplot as plt
from evaluate_performance import evaluate2

evaluate2(
    '../dataset/dataset.csv', # pubchem
    '../dataset/dataset_test_expanded.csv', # herbal db manual docking
    '03-evaluate-3_result.csv',
    '03-evaluate-3_roc_chart.png',
    'ROC on Herbal DB Dataset manual docking',
    'PubChem BioAssay + DUD-E as training data',
    '03-evaluate-3_scores_chart.png',
    'Performance Comparison on Herbal DB manual docking',
    'PubChem BioAssay + DUD-E as training data'
)

# show plots
plt.show()
