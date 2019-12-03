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
print(df_train_complete.groupby("Interest").mean())
#plt.matshow(df_train_complete)

#--------------------- RFE features for FTG------------
target_value="Interest"

y = df_train_y[target_value]
X = df_train_x
#no of features
nof_list=np.arange(1,13)            
high_score=0
#Variable to store the optimum features
nof=0           
score_list =[]
for n in range(len(nof_list)):
    X_train, X_test, y_train, y_test = train_test_split(df_train_x,y, test_size = 0.3, random_state = 0)
    model = LinearRegression()
    rfe = RFE(model,nof_list[n])
    X_train_rfe = rfe.fit_transform(X_train,y_train)
    X_test_rfe = rfe.transform(X_test)
    model.fit(X_train_rfe,y_train)
    score = model.score(X_test_rfe,y_test)
    score_list.append(score)
    if(score>high_score):
        high_score = score
        nof = nof_list[n]
print("Optimum number of features: %d" %nof)
print("Score with %d features: %f" % (nof, high_score))

cols = list(df_train_x.columns)
model = LinearRegression()
#Initializing RFE model
rfe = RFE(model, nof)             
#Transforming data using RFE
X_rfe = rfe.fit_transform(X,y)  
#Fitting the data to model
model.fit(X_rfe,y)              
temp = pd.Series(rfe.support_,index = cols)
selected_features_rfe = temp[temp==True].index
print("RFE SELECTED FEATURES", selected_features_rfe)

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

#---------------------PCA Components------------
scaled_df = StandardScaler().fit_transform(df_train_x[selected_features_rfe])
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(scaled_df)
print("Explained variance",pca.explained_variance_)
print("Explained variance ratio",pca.explained_variance_ratio_)
print("Explained components",pca.n_features_)
print("Explained components _",pca.n_components_)
pca_df = pd.DataFrame(data=principalComponents, columns=['PCA1', 'PCA2'])
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.scatter(pca_df["PCA1"], pca_df["PCA2"])
plt.show()
pca_data = pca_df["PCA1"].values.reshape(-1,1)
linear_regressor = LinearRegression()  # create object for the class
linear_regressor.fit(pca_data, df_train_y["FTG"])  # perform linear regression
Y_pred = linear_regressor.predict(pca_data)  # make predictions
plt.scatter(pca_data, df_train_y["FTG"])
plt.plot(pca_data, Y_pred, color='red')
plt.show()

#---------------------Linear Regression 0 ------------
print()
print("Linear regression")
X = df_train_x[selected_features_BE]
x_test = df_test_x[selected_features_BE]
y = df_train_y[["FTG"]]
reg = LinearRegression().fit(X,y)
print("Score", reg.score(X, y))
print("Coef", reg.coef_)
y_pred = reg.predict(x_test)
plt.plot(y_pred-df_test_y[["FTG"]].values)
#plt.plot(x_test["HomeTeam"], y_pred, color="red")
#plt.xticks(())
#plt.yticks(())
plt.show()

#---------------------Linear Regression------------
r_squared = {}
for name in selected_features_rfe: #df_train_x.columns:
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
#for key in r_squared_ordered:
#    print(str(key) + ": " + str(r_squared[key]))

X = df_train_x[max(r_squared.items(), key=operator.itemgetter(1))[0]].values.reshape(-1,1) ## X usually means our input variables (or independent variables)
y = df_train_y["FTG"] ## Y usually means our output/dependent variable

linear_regressor = LinearRegression()  # create object for the class
linear_regressor.fit(X, y)  # perform linear regression
Y_pred = linear_regressor.predict(X)  # make predictions
print("Test score: " , linear_regressor.score(df_test_x[max(r_squared.items(), key=operator.itemgetter(1))[0]].values.reshape(-1,1),df_test_y["FTG"]))

rr = Ridge(alpha=0.025)
rr.fit(df_train_x, df_train_y["FTG"])
Ridge_train_score = rr.score(df_train_x, df_train_y["FTG"])
Ridge_test_score = rr.score(df_test_x, df_test_y["FTG"])
print("ridge regression train score low alpha:", Ridge_train_score)
print("ridge regression test score low alpha:", Ridge_test_score)

# Print out the statistics
#print(model.summary())
#print(LinearRegression().fit(X, y).score(X, y))

#Tähän tarvii plottaa toi data scatterina ja sitten predictions linearinen
plt.scatter(X, y)
plt.plot(X, Y_pred, color='red')
plt.xlabel('HST')
plt.ylabel('FTG')
plt.show()



#--------------------------------------------------
#---------------------Correlation matrix features for FTG------------
cor = df_train_complete.corr()
np.fill_diagonal(cor.values, np.nan)
plt.matshow(cor)
#plt.show()
cor_target = abs(cor[target_value])
relevant_features = cor_target[cor_target>0.25]
print("Relevant features by correlation matrix")
print(relevant_features)



#--------------------- Embeded method features for FTG------------

y = df_train_y[target_value]
X = df_train_x
reg = LassoCV()
reg.fit(X, y)
print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" %reg.score(X,y))
coef = pd.Series(reg.coef_, index = X.columns)

print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")

imp_coef = coef.sort_values()
import matplotlib
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Feature importance using Lasso Model")
#plt.show()
#---------------------Logarithmic Regression------------
os = SMOTE(random_state=0)
columns = df_train_x.columns
os_data_X,os_data_y=os.fit_sample(df_train_x, df_train_y["Interest"])
os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
os_data_y= pd.DataFrame(data=os_data_y,columns=['y'])
# we can Check the numbers of our data
print("length of oversampled data is ",len(os_data_X))
print("Number of none interesting matches in oversampled data",len(os_data_y[os_data_y['y']==0]))
print("Number of interesting matches",len(os_data_y[os_data_y['y']==1]))
print("Proportion of no subscription data in oversampled data is ",len(os_data_y[os_data_y['y']==0])/len(os_data_X))
print("Proportion of subscription data in oversampled data is ",len(os_data_y[os_data_y['y']==1])/len(os_data_X))

logreg = LogisticRegression(solver="lbfgs", max_iter=9999)

rfe = RFE(logreg, 5)
rfe = rfe.fit(os_data_X, os_data_y.values.ravel())
print(rfe.support_)
print(rfe.ranking_)
print((df_train_x.T[rfe.support_]).T)

df_train_x_rfe = df_train_x.T[rfe.support_].T.drop(columns=['HTR'])
df_test_x_rfe = df_test_x.T[rfe.support_].T.drop(columns=['HTR'])

logit_model=sm.Logit(df_train_y["Interest"],df_train_x_rfe)
result=logit_model.fit()
print(result.summary2())

logreg.fit(df_train_x_rfe, df_train_y["Interest"])
y_pred = logreg.predict(df_test_x_rfe)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(df_test_x_rfe, df_test_y["Interest"])))
confusion_matrix = confusion_matrix(df_test_y["Interest"], y_pred)
print(confusion_matrix)
#------------------------------------------------------