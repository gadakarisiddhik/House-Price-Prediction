import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix


# Dataset ko CSV file se load kar rahe hain
housing = pd.read_csv("dataset/boston.csv")


# Dataset ki starting ki 5 rows dekhne ke liye
# print(housing.head())

print("\n")


# Dataset ki last ki 5 rows dekhne ke liye
# print(housing.tail())

print("\n")


# Dataset ki basic information dekhne ke liye
# Isse columns, data types aur missing values ka idea milta hai
# print(housing.info())


# Dataset ke important columns:
# crim  -> Crime rate
# zn    -> Residential land proportion
# indus -> Non-retail business area
# chas  -> Charles River ke paas hai ya nahi
# nox   -> Nitric oxide concentration
# rm    -> Average number of rooms
# age   -> Old houses ka percentage
# dis   -> Employment centres se distance
# rad   -> Highway accessibility
# tax   -> Property tax rate
# ptratio -> Pupil-teacher ratio
# b     -> Black population related index
# lstat -> Lower status population percentage
# medv  -> House ki median value


print("\n")


# Dataset ka statistical summary dekhne ke liye
# Isme mean, standard deviation, minimum, maximum etc. milta hai
# print(housing.describe())

print("\n")


# Dataset ke har numerical column ka histogram banate hain
# Histogram se data ka distribution samajhne me help milti hai
# print(housing.hist(bins=50, figsize=(12, 15)))


# Histogram ko screen par show karne ke liye
# print(plt.show())


# ============================================================
# Train Test Spliting
# ============================================================


"""
--------------------for learnig perpus-------------------

# Ye function manually train aur test dataset banane ke liye tha
# Sirf learning purpose ke liye rakha gaya hai

def split_train_test(data, test_ration):

    # Same random result baar-baar lane ke liye seed set kar rahe hain
    np.random.seed(42)

    # Dataset ke indexes ko randomly shuffle kar rahe hain
    shuffled = np.random.permutation(len(data))

    # Test dataset ka size calculate kar rahe hain
    test_set_size = int(len(data) * test_ration)

    # Starting ke indexes test dataset ke liye
    test_indieces = shuffled[:test_set_size]

    # Baaki indexes training dataset ke liye
    train_indieces = shuffled[test_set_size:]

    # Training aur testing dataset return kar rahe hain
    return data.iloc[train_indieces], data.iloc[test_indieces]


# Function ko use karke train aur test dataset bana rahe hain
train_set, test_set = split_train_test(
    housing,
    0.2
)


# Train aur test dataset ki rows count check kar rahe hain
print(
    f"Rows in train set: {len(train_set)}\n"
    f"Rows in test set: {len(test_set)}"
)

"""


# Dataset ko training aur testing parts me divide kar rahe hain
# test_size=0.2 ka matlab 20% testing aur 80% training
# random_state=42 se same split baar-baar milta hai
train_set, test_set = train_test_split(
    housing,
    test_size=0.2,
    random_state=42
)


# Training aur testing dataset me kitni rows hain wo check kar rahe hain
print(
    f"Rows in train set: {len(train_set)}\n"
    f"Rows in test set: {len(test_set)}"
)


# ============================================================
# Stratified Train Test Split
# ============================================================

# StratifiedShuffleSplit ka use karke
# kisi particular column ka proportion maintain karte hain
split = StratifiedShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)


# "chas" column ke basis par stratified splitting kar rahe hain
for train_index, test_index in split.split(housing, housing["chas"]):

    # Training dataset ke indexes select kar rahe hain
    strat_train_set = housing.loc[train_index]

    # Testing dataset ke indexes select kar rahe hain
    strat_test_set = housing.loc[test_index]


# Test dataset me chas ke values ka count check karne ke liye
# print(strat_test_set["chas"].value_counts())


# Training dataset me chas ke values ka count check karne ke liye
# print(strat_train_set["chas"].value_counts())


# ============================================================
# Looking for Correlations
# ============================================================


# Sabhi numerical columns ke beech correlation calculate kar rahe hain
corr_matrix = housing.corr()


# "medv" ke saath baaki columns ka correlation dekhne ke liye
# Highest correlation se lowest correlation tak sort kar rahe hain
# print(corr_matrix["medv"].sort_values(ascending=False))


# Scatter matrix ke liye important attributes select kar rahe hain
attributes = ["medv", "rm", "zn", "lstat"]


# Selected columns ke beech relationship visualize karne ke liye
print(scatter_matrix(housing[attributes], figsize=(12, 8)))


# Video ka time reference
# time : 1:26:09


# ============================================================
# RM vs MEDV Scatter Plot
# ============================================================

# RM = average number of rooms
# MEDV = house ki median value
# Ye graph rooms aur house price ke relationship ko show karega
#
# alpha=0.1 ka matlab points ko thoda transparent rakha hai
# taaki overlapping points clearly dikh sake
print(
    housing.plot(
        kind="scatter",
        x="rm",
        y="medv",
        alpha=0.1
    )
)
plt.show()