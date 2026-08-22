import pandas as pd
import numpy as pn

housing = pd.read_csv("dataset/boston.csv")
print(housing.head())
print("\n")
print(housing.tail())
print("\n")
print(housing.info())
print("\n")
print(housing.describe())