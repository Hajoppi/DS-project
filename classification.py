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
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from collections import OrderedDict
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso

df_train_x = pd.read_csv('football_train_x.csv')
df_train_y = pd.read_csv('football_train_y.csv')
df_test_x = pd.read_csv('football_test_x.csv')
df_test_y = pd.read_csv('football_test_y.csv')

df_train_complete = pd.concat([df_train_x,df_train_y],axis = 1)

target_value="Interest"

#---------------------Backward Elimination features for FTG------------
X_1 = sm.add_constant(df_train_x)
y = df_train_y[target_value]
model = sm.OLS(y,X_1).fit()
model.pvalues

cols = list(df_train_x.columns)
pmax = 1
while (len(cols)>0):
    p= []
    X_1 = df_train_x[cols]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(y,X_1).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break

selected_features_BE = cols
print("Selected features with backward propagations")
print(selected_features_BE)

#---------------------Logarithmic Regression------------
logreg = LogisticRegression(solver="lbfgs", max_iter=9999)

df_train_x_rfe = df_train_x[selected_features_BE]
df_test_x_rfe = df_test_x[selected_features_BE]

logit_model=sm.Logit(df_train_y["Interest"],df_train_x_rfe)
result=logit_model.fit()
print(result.summary2())

logreg.fit(df_train_x_rfe, df_train_y["Interest"])
y_pred = logreg.predict(df_test_x_rfe)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(df_test_x_rfe, df_test_y["Interest"])))
confusion_matrix = confusion_matrix(df_test_y["Interest"], y_pred)
print(confusion_matrix)
labels=["1","0"]
sns.heatmap(confusion_matrix, square=True, annot=True, fmt='d', cbar=False, xticklabels=labels, yticklabels=labels)
plt.yticks(range(3))
plt.yticks(rotation=30)
plt.ylabel('predicted label')
plt.xlabel('true label')
plt.title('Interest Confusion Matrix')
plt.show()