import os
import pandas as pd 
import numpy as np
import random
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

df = pd.read_csv(r"C:\Users\manas\laststraw\templates\diabetes_012_health_indicators_BRFSS2015.csv", header = None, skiprows = 1)

useDf = df.drop([4, 9, 14, 13, 16, 20, 21], axis = 1)
print(useDf)
#useDf = useDf.rename(columns={'Diabetes_012': 'DiabetesPrediction', 'DiffWalk': 'Walking'})

ClassModel = DecisionTreeClassifier()
features = useDf.loc[:, useDf.columns != 0]
labels = useDf[0]
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.10)
ClassModel.fit(features_train, labels_train)

#array = np.array([[1.0, 1.0, 28.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 22.0, 0.0, 0.0, 0.0, 0.0]])

train_predictions = ClassModel.predict(features_train)
#print(features_train)
#test_predictions = ClassModel.predict(array)
#print(test_predictions)

pickle.dump(ClassModel, open('model.pkl','wb'))
