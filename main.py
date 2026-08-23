import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix

housing = pd.read_csv("dataset/boston.csv")
print(housing.head())
print("\n")
print(housing.tail())
print("\n")
print(housing.info())
# crim	zn	indus	chas	nox	rm	age	dis	rad	tax	ptratio	b	lstat	medv
print("\n")
print(housing.describe())
print("\n")

print(housing.hist(bins=50, figsize=(12, 15)))
# print(plt.show())

# Train Test Spliting

"""
--------------------for learnig perpus-------------------

def split_train_test(data, test_ration):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ration)
    test_indieces = shuffled[:test_set_size]
    train_indieces = shuffled[test_set_size:]
    return data.iloc[train_indieces], data.iloc[test_indieces]

train_set, test_set = split_train_test(
    housing,
    0.2
)
print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}")

"""

train_set, test_set = train_test_split(
    housing,
    test_size=0.2,
    random_state=42
)
print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}")

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['chas']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

print(strat_test_set['chas'].value_counts())
print(strat_train_set['chas'].value_counts())

# Looking for Correlations
corr_matrix = housing.corr()
print(corr_matrix['medv'].sort_values(ascending=False))

arrributes = ['medv','rm','zn','lstat']
print(scatter_matrix(housing[arrributes], figsize=(12, 8)))

# time : 1:26:09 