#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Follwing this tutorial for tuning an xgboost model
https://towardsdatascience.com/fine-tuning-xgboost-in-python-like-a-boss-b4543ed8b1e
"""
#%%
#Import libraries:
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier

#Additional scklearn functions
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV  
from sklearn import metrics 
from sklearn.metrics import accuracy_score

  
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

combined_data = pickle.load(open("../pickled_corpus_w_features2.sav", 'rb'))
final_col_order = combined_data.columns.tolist()

#target = 'source'
#IDcol = 'text' # not sure if this will be correct

#%%
# split data into X and y
X = combined_data.drop(["text", "source"], axis = 1)
Y = combined_data['source'] # "source" is the column of numeric sources

col_names = X.columns

# split data into train and test sets
seed = 10
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

# This time we will scale the data correctly
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test) 

# Make the split data back into a dataframe
#X_train = pd.DataFrame(X_train, columns = col_names)
#X_test = pd.DataFrame(X_test, columns = col_names)

model = XGBClassifier(n_estimators = 200)
#model.fit(X_train, y_train) 

eval_set = [(X_train, y_train), (X_test, y_test)]
eval_metric = ["merror"]
model.fit(X_train, y_train,
          eval_metric=eval_metric, 
          eval_set=eval_set, 
          verbose=True)

train_merror = model.evals_result()['validation_0']['merror']
test_merror = model.evals_result()['validation_1']['merror']
merror_df = pd.DataFrame({'train': train_merror, 
                          'test': test_merror,
                          'iteration': range(200)})

y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0)) 


fig = plt.figure(figsize=(5,5))
plt.plot( 'iteration', 'train', data=merror_df, marker='', color='olive', linewidth=2, label="train")
plt.plot( 'iteration', 'test', data=merror_df, marker='', color='olive', linewidth=2, linestyle='dashed', label="test")
plt.legend()