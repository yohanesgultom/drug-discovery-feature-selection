"""
Visualize dataset:
* Feature importances using Extra Trees
* 2D plot using t-SNE with various perplexities

Execution time:
real    15m50.854s
user    14m30.858s
sys     0m30.375s

@author yohanes.gultom@gmail.com
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas
import os
from matplotlib.ticker import NullFormatter
from sklearn import manifold, datasets
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import ExtraTreesClassifier
from time import time

# config
dataset_file = '../dataset/dataset.csv'
n_components = 2
perplexities = [5, 30, 50, 100]
chart_importances_filename = 'visualize-dataset_importances.png'
chart_filename_tpl = 'visualize-dataset_tsne_{}.png'

# check if display available
if os.name == 'posix' and "DISPLAY" not in os.environ:
    matplotlib.use('Agg')

# read dataset
df = pandas.read_csv(dataset_file, index_col=0)
feature_names = list(df[df.columns.drop('Class')])

# split to data X and labels y
X = df[df.columns.drop('Class')].values.astype('float32')
y = df['Class'].values

# separate data by class
red = y == 0
green = y == 1

# scale data
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# feature importance check using Extra Trees
forest = ExtraTreesClassifier(n_estimators=250, random_state=0)
forest.fit(X, y)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1] # reverse
n = 10
print("Top {} most important features:".format(n))
for f in range(min(X.shape[1], n)):
    print("{}. feature {} ({}): {:.4g}".format(f + 1, indices[f], feature_names[indices[f]], importances[indices[f]]))

# Set figure size to 1200 x 880 px
plt.figure(figsize=(15, 11))

# Plot the feature importances of the forest
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances, color="r", align="center")
plt.xlim([-1, X.shape[1]])
plt.ylabel("Importance")
plt.xlabel("Feature Index")
plt.savefig(chart_importances_filename)

# visualize dataset with TSNE
for i, perplexity in enumerate(perplexities):
    t0 = time()
    tsne = manifold.TSNE(n_components=n_components, init='random', random_state=0, perplexity=perplexity)
    Y = tsne.fit_transform(X)
    t1 = time()
    print("t-SNE perplexity={} in {:.2g} sec".format(perplexity, t1 - t0))

    # plot
    fig, ax = plt.subplots() 
    ax.set_title("Perplexity=%d" % perplexity)
    ax.scatter(Y[red, 0], Y[red, 1], c="r")
    ax.scatter(Y[green, 0], Y[green, 1], c="g")
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    ax.axis('tight')
    filename = chart_filename_tpl.format(perplexity)
    plt.savefig(filename)
    print("chart saved in {}".format(filename))

plt.show()