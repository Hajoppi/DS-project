import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import operator

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
from imblearn.over_sampling import SMOTE
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from collections import OrderedDict


df_train_x = pd.read_csv('football_train_x.csv')
df_train_y = pd.read_csv('football_train_y.csv')
df_test_x = pd.read_csv('football_test_x.csv')
df_test_y = pd.read_csv('football_test_y.csv')
#print(df_train_x)
df_train_complete = pd.concat([df_train_x,df_train_y],axis = 1)
print(df_train_complete.groupby("Interest").mean())
plt.matshow(df_train_complete)

#---------------------Linear Regression------------
r_squared = {}
for name in df_train_x.columns:
    X = df_train_x[name]
    y = df_train_y['FTG']

    # Note the difference in argument order
    model = sm.OLS(y, X).fit()
    predictions = model.predict(X) # make the predictions by the model

    # Save R^2 in vaiable
    r_squared[name] = model.rsquared

    # Print out the statistics
    #print(model.summary())
    r_squared_ordered = OrderedDict(sorted(r_squared.items(), key=lambda x: x[1]))
for key in r_squared_ordered:
    print(str(key) + ": " + str(r_squared[key]))

X = df_train_x[max(r_squared.items(), key=operator.itemgetter(1))[0]] ## X usually means our input variables (or independent variables)
y = df_train_y["Intercept"] ## Y usually means our output/dependent variable
X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model

# Note the difference in argument order
model = sm.OLS(y, X).fit() ## sm.OLS(output, input)
predictions = model.predict(X)

# Print out the statistics
print(model.summary())

#--------------------------------------------------

#---------------------Logarithmic Regression------------
os = SMOTE(random_state=0)
columns =df_train_x.columns
os_data_X,os_data_y=os.fit_sample(df_train_x, df_train_y["Interest"])
os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
os_data_y= pd.DataFrame(data=os_data_y,columns=['y'])
# we can Check the numbers of our data
print("length of oversampled data is ",len(os_data_X))
print("Number of none interesting matches in oversampled data",len(os_data_y[os_data_y['y']==0]))
print("Number of interesting matches",len(os_data_y[os_data_y['y']==1]))
print("Proportion of no subscription data in oversampled data is ",len(os_data_y[os_data_y['y']==0])/len(os_data_X))
print("Proportion of subscription data in oversampled data is ",len(os_data_y[os_data_y['y']==1])/len(os_data_X))

logreg = LogisticRegression()

rfe = RFE(logreg, 5)
rfe = rfe.fit(os_data_X, os_data_y.values.ravel())
#print(rfe.support_)
#print(rfe.ranking_)
#print((df_train_x.T[rfe.support_]).T)
df_train_x_rfe = (df_train_x.T[rfe.support_]).T
df_test_x_rfe = (df_test_x.T[rfe.support_]).T.drop(columns=['HTR'])

logit_model=sm.Logit(df_train_y["Interest"],df_train_x_rfe)
result=logit_model.fit()
print(result.summary2())

logreg.fit(df_train_x_rfe, df_train_y["Interest"])
y_pred = logreg.predict(df_test_x_rfe)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(df_test_x_rfe, df_test_y["Interest"])))
confusion_matrix = confusion_matrix(df_test_y["Interest"], y_pred)
print(confusion_matrix)
#------------------------------------------------------