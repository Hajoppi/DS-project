import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
import seaborn as sns

df_train_x = pd.read_csv('football_train_x.csv')
df_train_y = pd.read_csv('football_train_y.csv')
df_test_x = pd.read_csv('football_test_x.csv')
df_test_y = pd.read_csv('football_test_y.csv')

asdf = LogisticRegression.fit(df_train_x, df_train_y)
