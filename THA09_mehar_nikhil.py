import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn import metrics

#Neural Network
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

#Multiple Regression
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as smf

data=pd.read_table('CaliforniaHospital_FinancialData.txt', sep ='\t')
data.head()
data.columns

data1=data[['QRT','NET_TOT']]
data2=data[['QRT','TOT_OP_EXP']]
data3=data[['QRT','NONOP_REV']]

s1=data1.NET_TOT


#Lag 1 Effect
lag1col_nt= pd.Series([0])
lag1col_nt= lag1col_nt.append(s1)
lag1col_nt= lag1col_nt.reset_index(drop=True)
lag1col_nt= lag1col_nt.ix[0:38,]

lag2col_nt = pd.Series([0,0])
lag2col_nt = lag2col_nt.append(s1)
lag2col_nt = lag2col_nt.reset_index(drop=True)
lag2col_nt = lag2col_nt.ix[0:38,]



#=======================================
# Step 3: Add data back into dataframe
#=======================================

newcols1 = pd.DataFrame({'lag1': lag1col_nt})
nettot_data2 = pd.concat([data1, newcols1], axis=1)

newcols2 = pd.DataFrame({'lag2': lag2col_nt})
nettot_data3 = pd.concat([nettot_data2, newcols2], axis=1)

nettot_data3 = nettot_data3[['NET_TOT','lag1','lag2','QRT']]


#=======================
# Create Time Variable
#=======================
timelen = len(nettot_data3.index) + 1
newcols3 = pd.DataFrame({'time': list(range(1,timelen))})
nettot_data4 = pd.concat([nettot_data3, newcols3], axis=1)

#Finalized data with 2 lag effects
nettot_data5 = nettot_data4[['NET_TOT','time','lag1','lag2']]

#=====================================
# Data splitting for time series. Do
# not randomly pull data! The data
# must be linear and incremental.
#=====================================
splitnum = np.round((len(nettot_data5.index) * 0.7), 0).astype(int)
splitnum
#### Result: 70% of data includes 90 records

nettot_train = nettot_data5.ix[0:27,]
nettot_test = nettot_data5.ix[27:38,]


nnts1 = MLPRegressor(activation='relu', solver='sgd')
nnts1.fit(nettot_train[['time','lag1','lag2']], nettot_train.NET_TOT)

nnts1_pred = nnts1.predict(nettot_test[['time','lag1','lag2']])

metrics.mean_absolute_error(nettot_test.NET_TOT, nnts1_pred)

metrics.mean_squared_error(nettot_test.NET_TOT, nnts1_pred)