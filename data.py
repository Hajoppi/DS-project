import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
from imblearn.over_sampling import SMOTE
import statsmodels.api as sm

df_train_x = pd.read_csv('football_train_x.csv')
df_train_y = pd.read_csv('football_train_y.csv')
df_test_x = pd.read_csv('football_test_x.csv')
df_test_y = pd.read_csv('football_test_y.csv')
#print(df_train_x)
df_train_complete = pd.concat([df_train_x,df_train_y],axis = 1)
print(df_train_complete.groupby("Interest").mean())



#---------------------
os = SMOTE(random_state=0)
columns =df_train_x.columns
os_data_X,os_data_y=os.fit_sample(df_train_x, df_train_y["Interest"])
os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
os_data_y= pd.DataFrame(data=os_data_y,columns=['y'])
# we can Check the numbers of our data
print("length of oversampled data is ",len(os_data_X))
print("Number of no subscription in oversampled data",len(os_data_y[os_data_y['y']==0]))
print("Number of subscription",len(os_data_y[os_data_y['y']==1]))
print("Proportion of no subscription data in oversampled data is ",len(os_data_y[os_data_y['y']==0])/len(os_data_X))
print("Proportion of subscription data in oversampled data is ",len(os_data_y[os_data_y['y']==1])/len(os_data_X))
#---------------------


logreg = LogisticRegression()

rfe = RFE(logreg, 20)
rfe = rfe.fit(os_data_X, os_data_y.values.ravel())
print(rfe.support_)
print(rfe.ranking_)


logit_model=sm.Logit(df_train_y,df_train_x)
result=logit_model.fit()
print(result.summary2())