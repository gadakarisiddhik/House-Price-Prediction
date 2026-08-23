import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
def split_train_test(data, test_ration):
    shuffled = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ration)
    test_indieces = shuffled[:test_set_size]
    train_indieces = shuffled[test_set_size:]
    return data.iloc[train_indieces], data.iloc[test_indieces]