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
#print(df_train_x)
df_train_complete = pd.concat([df_train_x,df_train_y],axis = 1)
#sns.pairplot(df_train_x.drop(['HomeTeam', 'AwayTeam'], axis = 1))

plt.scatter(df_train_complete[['HS']],df_train_complete[['AS']])
plt.xlabel('HS')
plt.ylabel('AS')
plt.show()

#---------------------PCA Components------------
scaled_df = StandardScaler().fit_transform(df_train_x)
pca = PCA(n_components=10)
principalComponents = pca.fit_transform(scaled_df)
print("Explained variance",pca.explained_variance_)
print("Explained variance ratio",pca.explained_variance_ratio_)
#print("Explained components",pca.n_features_)
#print("Explained components _",pca.n_components_)
#pca_df = pd.DataFrame(data=principalComponents, columns=['PCA1', 'PCA2'])
#plt.xlabel('PCA1')
#plt.ylabel('PCA2')
#plt.scatter(pca_df["PCA1"], pca_df["PCA2"])
#plt.show()
#pca_data = pca_df["PCA1"].values.reshape(-1,1)
#linear_regressor = LinearRegression()  # create object for the class
#linear_regressor.fit(pca_data, df_train_y["FTG"])  # perform linear regression
#Y_pred = linear_regressor.predict(pca_data)  # make predictions
#plt.scatter(pca_data, df_train_y["FTG"])
#plt.plot(pca_data, Y_pred, color='red')
#plt.show()

#pca = PCA().fit(digits.data)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');
plt.show()