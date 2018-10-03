import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 
from sklearn.cross_validation import KFold

os.chdir("F:\\r.python\\hw assignments\\msis5223_hwassignments\\msis5223_hwassignments")
os.getcwd

hosp_data=pd.read_csv("CaliforniaHospitalData.csv",sep =",")
hosp_data

emp_data=pd.read_table("CaliforniaHospitalData_Personnel.txt",sep ="\t")
emp_data

os.getcwd
cal_data=emp_data.merge(hosp_data,left_on='HospitalID',right_on='HospitalID')
cal_data

cal_data.columns
cal_data.shape

cal_data.drop(['Work_ID', 'PositionID', 'Website'],axis=1, inplace=True)
cal_data.columns

cal_data.shape
len(cal_data.index)
len(cal_data.columns)

hospitals=cal_data[(cal_data.Teaching=="Small/Rural") & (cal_data.AvlBeds>=15) & (cal_data.OperInc>=0)]
hospitals
hospitals.shape



hospitals.to_csv("hospital_data_new.txt", sep=",")

h_data=pd.read_table("hospital_data_new.txt", sep=",")
h_data

h_data.columns
h_data.rename(columns={'NoFTE':'FullTimeCount'}, inplace=True)
h_data.rename(columns={'NetPatRev':'NetPatientRevenue'}, inplace=True)
h_data.rename(columns={'InOperExp':'InpatientOperExp'}, inplace=True)
h_data.rename(columns={'OutOperExp':'OutpatientOperExp'}, inplace=True)
h_data.rename(columns={'OperRev':'Operating_Revenue'}, inplace=True)
h_data.rename(columns={'OperInc':'Operating_Income'}, inplace=True)
h_data.columns
h_data.dtypes
newrows=pd.DataFrame({'HospitalID':[20266,37393],
                       'Name':['Sonora Regional Medical Center - greenley','Barstow Community Hospital'],
                       'LastName':['Mehar','Mehar'],
                       'FirstName':['Nikhil','Nikhil'],
                       'Gender':['M','M'],
                       'PositionTitle':['Regional Representative','Regional Representative'],
                       'Compensation':[46978,46978],
                       'MaxTerm':[4,4],
                       'StartDate':['1/29/2017','1/29/2017'],
                       'Zip':['95370','92311'],
                       'TypeControl':['Non Profit','Investor'],
                       'Teaching':['Small/Rural','Small/Rural'],
                       'DonorType':['Charity','Charity']},
                     index=[28,29])
h_data2=pd.concat([h_data,newrows])
h_data2


h_data2.dtypes

h_data2['StartDate'] = pd.to_datetime(h_data2['StartDate'])
h_data2.StartDate.head()

richrr=h_data2[(h_data2.Operating_Income>=1000000)&(h_data2.PositionTitle=="Regional Representative")]
richrr

richrr.to_csv("richrr.txt", sep=",")

cal1_data=cal_data[(cal_data.TypeControl=="Non Profit") & ((cal_data.NoFTE>=250)|(cal_data.NetPatRev>=109000))]
cal1_data

cal1_data.to_csv("CALI1.txt", sep ="\t")

kf = KFold(len(cal_data.index), n_folds=4)
for train, test in kf:
    print("%s %s" % (train, test))

cal_data.ix[train]
cal_data.ix[test]

cal_data.ix[train].to_csv("training_data.txt", sep =",")
cal_data.ix[test].to_csv("testing_data.txt", sep =",")