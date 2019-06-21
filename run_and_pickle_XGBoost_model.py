#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 10:16:11 2019

@author: amandakomuro
"""
import pandas as pd
import numpy as np
import pickle
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
#%%
"""
Load pickled dataframe.
"""
#%% 
combined_data = pickle.load(open("./data/pickled_corpus_w_features.sav", 'rb'))
combined_data = combined_data.drop(["text_standard"], axis=1)
combined_data = combined_data.reset_index(drop=True)
final_col_order = combined_data.columns.tolist()
#%%
"""
Test that model still performs the same as during prototyping. Still getting 77.83%
"""
#%% 
# split data into X and y
X = combined_data.iloc[:,3:]
Y = combined_data['source'] # "source" is the column of numeric sources

# split data into train and test sets
seed = 8
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

# This time we will scale the data correctly
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train) #index now starts at 0
X_test = scaler.transform(X_test) #index now starts at 0

# fit model to training data
y_train2 = y_train.reset_index(drop=True)
y_test2 = y_test.reset_index(drop=True)

model = XGBClassifier()
model.fit(X_train, y_train2) # should I reset the index on y_train?

# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

# evaluate predictions
accuracy = accuracy_score(y_test2, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

#%%
"""
Calculate average scaled values for each feature and category. Will use for 
comparison in the web app when making suggestions.
"""
#%% 

labeled_scaled_df = pd.DataFrame(X_train)
labeled_scaled_df["source"] = y_train2
source_averages = labeled_scaled_df.groupby('source').mean()

source_averages.columns = final_col_order[3:]
source_averages = source_averages.reset_index(drop=True)

#%%
"""
Pickle the model for import in web app
"""
#%% 
filename = './insight2019/flask_app/my_flask/model/finalized_XGBoost_model.sav'
pickle.dump(model, open(filename, 'wb'))

filename = './insight2019/flask_app/my_flask/model/finalized_XGBoost_scaler.sav'
pickle.dump(scaler, open(filename, 'wb'))

filename = './insight2019/flask_app/my_flask/model/finalized_column_order.sav'
pickle.dump(final_col_order, open(filename, 'wb'))

filename = './insight2019/flask_app/my_flask/model/finalized_source_averages.sav'
pickle.dump(source_averages, open(filename, 'wb'))


#%%
"""
Confusion matrix, for presentations.
"""
#%% 
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = unique_labels(y_true)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


np.set_printoptions(precision=2)

# Plot normalized confusion matrix
plot_confusion_matrix(y_test, y_pred, classes = y_test, normalize=True,
                      title='Normalized confusion matrix')


#%%
"""
Validation curve - am I overfitting??
Tutorial adapted from: https://chrisalbon.com/machine_learning/model_evaluation/plot_the_validation_curve/
"""
#%%
#
#from sklearn.model_selection import validation_curve
#
## Create range of values for parameter
#param_range = np.arange(1, 250, 2)
#
## Calculate accuracy on training and test set using range of parameter values
#train_scores, test_scores = validation_curve(XGBClassifier(), 
#                                             X, 
#                                             Y, 
#                                             param_name="n_estimators", 
#                                             param_range=param_range,
#                                             cv=3, 
#                                             scoring="accuracy", 
#                                             n_jobs=-1)
#
#
## Calculate mean and standard deviation for training set scores
#train_mean = np.mean(train_scores, axis=1)
#train_std = np.std(train_scores, axis=1)
#
## Calculate mean and standard deviation for test set scores
#test_mean = np.mean(test_scores, axis=1)
#test_std = np.std(test_scores, axis=1)
#
## Plot mean accuracy scores for training and test sets
#plt.plot(param_range, train_mean, label="Training score", color="black")
#plt.plot(param_range, test_mean, label="Cross-validation score", color="dimgrey")
#
## Plot accurancy bands for training and test sets
#plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, color="gray")
#plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, color="gainsboro")
#
## Create plot
#plt.title("Validation Curve With Random Forest")
#plt.xlabel("Number Of Trees")
#plt.ylabel("Accuracy Score")
#plt.tight_layout()
#plt.legend(loc="best")
#plt.show()


