from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import sklearn.metrics as metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("C:/Users/Lochan/OneDrive/Documents/P5Data.csv")
X = dataset.iloc[:, :-1]  
label = {'Setosa': 0,'Versicolor': 1, 'Virginica': 2} 
y = [label[c] for c in dataset.iloc[:, -1]]
plt.figure(figsize=(14,7))
colormap=np.array(['red','lime','black'])
# REAL PLOT
plt.subplot(1,3,1)
plt.title('Real')
plt.scatter(X.petal_length,X.petal_width,c=colormap[y])
# K-PLOT
model=KMeans(n_clusters=3, random_state=3425).fit(X)
plt.subplot(1,3,2)
plt.title('KMeans')
plt.scatter(X.petal_length,X.petal_width,c=colormap[model.labels_])
print('The accuracy score of K-Mean: ',metrics.accuracy_score(y, model.labels_))
print('The Confusion matrixof K-Mean:\n',metrics.confusion_matrix(y, model.labels_))
# GMM PLOT
gmm=GaussianMixture(n_components=3, random_state=3425).fit(X)
y_cluster_gmm=gmm.predict(X)
plt.subplot(1,3,3)
plt.title('GMM Classification')
plt.scatter(X.petal_length,X.petal_width,c=colormap[y_cluster_gmm])
print('The accuracy score of EM: ',metrics.accuracy_score(y, y_cluster_gmm))
print('The Confusion matrix of EM:\n ',metrics.confusion_matrix(y, y_cluster_gmm))