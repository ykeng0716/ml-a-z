# -*- coding: utf-8 -*-

# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('Data.csv')

## [:, :-1] 1st : -> take all the lines | 2nd :-1 -> take all coluｑms exvcept last column
X = dataset.iloc[:, :-1].values 

## iloc[:, 3] all lines and 3rd column
y = dataset.iloc[:, 3].values 

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)

## X[:, 1:3] 1:3 -> take index 1(cloumn2), index 2(column 3)
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()

## for label of Farance,Spain to the digit number
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

## dummy Encoding 為了預防machine learing思考 Ｆrance > Gemmany, Gemmany > Spain
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

## for label of Yes, No
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


#  Feature Scaling
## 因為X, y的級距不同，所有要把兩種或多鐘單位的數值，放在同一量級裡
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X. transform(X_test)









