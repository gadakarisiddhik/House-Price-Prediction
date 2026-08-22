import pandas as pd
import numpy as pn
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