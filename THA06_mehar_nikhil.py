import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

#Binning of data
from scipy.stats import binned_statistic

#Regression output
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as smf


os.getcwd()
hos_data=pd.read_table('mehar_nikhil_export.txt')
hos_data.columns
hos_data.dtypes

hos_data.drop(['Name','LastName','FirstName','Work_ID','PositionID','StartDate','startdate'],axis=1,inplace=True)
hos_data.dtypes

pd.unique(hos_data.Teaching)
pd.unique(hos_data.DonorType)
pd.unique(hos_data.PositionTitle)
pd.unique(hos_data.Gender)
pd.unique(hos_data.Compensation)
pd.unique(hos_data.TypeControl)

hos_dummy1=pd.get_dummies(hos_data['Teaching'])
hos_dummy1.head()
hos_dummy1.columns= ['teach_s/r','teach_teaching']
hos_data=hos_data.join(hos_dummy1)

hos_dummy2=pd.get_dummies(hos_data['DonorType'])
hos_dummy2.head()
hos_dummy2.columns=['donor_charity','donor_alumni']
hos_data=hos_data.join(hos_dummy2)

hos_dummy3=pd.get_dummies(hos_data['PositionTitle'])
hos_dummy3.head()
hos_dummy3.columns=['post_actdir','post_regrep','post_sim','post_sbrep']
hos_data=hos_data.join(hos_dummy3)

hos_dummy4=pd.get_dummies(hos_data['Gender'])
hos_dummy4.head()
hos_dummy4.columns=['male','female']
hos_data=hos_data.join(hos_dummy4)

hos_dummy5=pd.get_dummies(hos_data['Compensation'])
hos_dummy5.head()
hos_dummy5.columns=['comp_249k','comp_47k','comp_24k','comp_89k']
hos_data=hos_data.join(hos_dummy5)

hos_dummy6=pd.get_dummies(hos_data['TypeControl'])
hos_dummy6.head()
hos_dummy6.columns=['control_city/county','control_district','control_investor','control_nonprofit']
hos_data=hos_data.join(hos_dummy6)



hos_data.AvlBeds.max()
hos_data.AvlBeds.min()
hos_data.AvlBeds.plot.hist(alpha=0.5)
bin_counts,bin_edges,binnum=binned_statistic(hos_data.AvlBeds,hos_data.AvlBeds,statistic='count',bins=6)

bin_counts
bin_edges

bin_counts,bin_edges,binnum=binned_statistic(hos_data.AvlBeds,hos_data.AvlBeds,statistic='count',bins=15)
bin_counts
bin_edges

bin_interval=[12,70,130,550,910]
bin_counts,bin_edges,binnum=binned_statistic(hos_data.AvlBeds,hos_data.AvlBeds,statistic='count',bins=bin_interval)
bin_counts
bin_edges

binlabels=['beds_12_70','beds_70_130','beds_130_550','beds_550_910']

avlbeds_categ=pd.cut(hos_data.AvlBeds,bin_interval,right=False,retbins=False, labels=binlabels)
avlbeds_categ.name = 'avlbeds_categ'


hos_data=hos_data.join(pd.DataFrame(avlbeds_categ))
hos_data[['AvlBeds', 'avlbeds_categ']].sort_values(by='AvlBeds')

hos_data.drop(['beds_categ'],axis=1,inplace=True)
hos_data.dtypes

hos_dummy7=pd.get_dummies(hos_data['avlbeds_categ'])
hos_dummy7.head()
hos_data=hos_data.join(hos_dummy7)

hos_reg1=smf.ols('OperInc~ teach_teaching+ donor_charity + post_actdir + post_regrep + post_sim + male + comp_47k + comp_24k + comp_89k + control_nonprofit + control_district + control_investor + beds_12_70 + beds_70_130 + beds_130_550',hos_data).fit()
hos_reg1.summary()

hos_reg2=smf.ols('OperRev~ teach_teaching+ donor_charity + post_actdir + post_regrep + post_sim + male + comp_47k + comp_24k + comp_89k + control_nonprofit + control_district + control_investor + beds_12_70 + beds_70_130 + beds_130_550',hos_data).fit()
hos_reg2.summary()

hos_reg3=smf.ols('OperInc~comp_89k + control_nonprofit + control_district + control_investor + post_actdir + post_sim',hos_data).fit()
hos_reg3.summary()