# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset = pd.read_csv('SI_Data_Deployment.csv')

X = dataset.iloc[:, :4]

y = dataset.iloc[:, -1]

from sklearn.linear_model import LogisticRegression
regressor = LogisticRegression()

regressor.fit(X,y)

pickle.dump(regressor, open('model.pkl', 'wb'))

model = pickle.load(open('model.pkl', 'rb'))
#Z = model.predict_proba([[1, 1, 1, 1]])
Z = model.predict_proba([[1, 1, 1, 1]])[0,1]
print(Z)

