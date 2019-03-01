from sklearn.cluster import KMeans
import numpy as np
import pandas

# config
raw_file = '../dataset/HerbalDB.csv'
top10_file = '../dataset/HerbalDB_labeled.csv'
n_clusters = 2
random_state = 40

def fillna(df):
    for col in df.columns[df.isna().any()].tolist():
        df[col].fillna(0, inplace = True)
    return df

# read dataset
df = pandas.read_csv(raw_file)
X = fillna(df.iloc[:, 1:]).values

df_test = pandas.read_csv(top10_file)
X_test = fillna(df_test.iloc[:, 1:]).values

clusterer = KMeans(n_clusters=n_clusters, random_state=random_state)
y_test = clusterer.fit_predict(X_test)

# y_test labels are expected to be true
acc = sum(y_test) / len(y_test) * 1.0
print('K-Means accuracy on top 10 data (Yanuar et al., 2014): {:.2f}'.format(acc))