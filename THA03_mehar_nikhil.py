import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.decomposition import PCA as pca
from sklearn.decomposition import FactorAnalysis as fact

#Clustering modules
import sklearn.metrics as metcs
from scipy.cluster import hierarchy as hier
from sklearn import cluster as cls

#For the tree
from sklearn.feature_extraction.image import grid_to_graph
from sklearn import tree
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus


data= pd.read_table('mehar_nikhil_export.txt')
data.columns

data
data.dtypes

data_pca=data[['NoFTE','NetPatRev','InOperExp', 'OutOperExp', 'OperRev', 'OperInc', 'AvlBeds','Compensation', 'MaxTerm','Work_ID','PositionID']]
pca_result = pca(n_components=11).fit(data_pca)

pca_result.explained_variance_

pca_result.components_

plt.figure(figsize=(7,5))
plt.plot([1,2,3,4,5,6,7,8,9,10,11], pca_result.explained_variance_ratio_, '-o')
plt.ylabel('Proportion of Variance Explained') 
plt.xlabel('Principal Component') 
plt.xlim(0.75,4.25) 
plt.ylim(0,1.05) 
plt.xticks([1,2,3,4,5,6,7,8,9,10,11])


data_fac= data[['NoFTE','NetPatRev','InOperExp', 'OutOperExp', 'OperRev', 'OperInc', 'AvlBeds','Compensation', 'MaxTerm','Work_ID','PositionID']]
fact_result = fact(n_components=3).fit(data_fac)

fact_result.components_


km = cls.KMeans(n_clusters=2).fit(data.loc[:,['NoFTE','AvlBeds']])
km.labels_      

data['Teaching'] = data['Teaching'].astype('category')          
data['TypeControl'] = data['TypeControl'].astype('category') 
data['DonorType'] = data['DonorType'].astype('category') 
rows, cols = data.shape

data.describe()


cm = metcs.confusion_matrix(data.Teaching, km.labels_)
print(cm) 

cm2 = metcs.confusion_matrix(data.TypeControl, km.labels_)
print(cm2) 

cm3 = metcs.confusion_matrix(data.DonorType, km.labels_)
print(cm3) 








