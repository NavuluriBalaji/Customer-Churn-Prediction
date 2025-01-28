import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
import pickle

te=pd.read_csv("C:\\Users\\digit\\Downloads\\customer.csv")
te1=te.head(n=1869)

te2=te1.drop(['CustomerID','Count','Country','State','City','Zip Code','Lat Long','Latitude','Longitude','Churn Value','Device Protection','Churn Score','Churn Reason','CLTV','Dependents'],axis=1)
te2['Total Charges']=pd.to_numeric(te2['Total Charges'])
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
te2['Gender']=le.fit_transform(te2['Gender'])
te2['Senior Citizen']=le.fit_transform(te2['Senior Citizen'])
te2['Phone Service']=le.fit_transform(te2['Phone Service'])
te2['Partner']=le.fit_transform(te2['Partner'])
te2['Multiple Lines']=le.fit_transform(te2['Multiple Lines'])
te2['Online Security']=le.fit_transform(te2['Online Security'])
te2['Online Backup']=le.fit_transform(te2['Online Backup'])
te2['Paperless Billing']=le.fit_transform(te2['Paperless Billing'])
te2['Churn Label']=le.fit_transform(te2['Churn Label'])
te2['Internet Service']=le.fit_transform(te2['Internet Service'])
te2['Contract']=le.fit_transform(te2['Contract'])
te2['Payment Method']=le.fit_transform(te2['Payment Method'])
te2['Tech Support']=le.fit_transform(te2['Tech Support'])
te2['Streaming TV']=le.fit_transform(te2['Streaming TV'])
te2['Streaming Movies']=le.fit_transform(te2['Streaming Movies'])

te3=te2.drop(['Senior Citizen','Multiple Lines','Streaming TV','Streaming Movies','Partner','Monthly Charges'],axis=1)
X=te3.drop(['Churn Label'],axis=1)
Y=te3['Churn Label']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=42)
from sklearn.svm import SVC
sc=SVC(kernel='poly')
sc.fit(X_train,Y_train)

pickle.dump(sc,open('model.pkl','wb'))