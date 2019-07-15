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
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

  
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

combined_data = pickle.load(open("../pickled_corpus_w_features3.sav", 'rb'))
final_col_order = combined_data.columns.tolist()

#target = 'source'
#IDcol = 'text' # not sure if this will be correct

#%%
"""
Split and prep the data.
"""
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
X_train = pd.DataFrame(X_train, columns = col_names)
X_test = pd.DataFrame(X_test, columns = col_names)
#%%
"""
1st parameter to optimize: n_estimators.
Other parameters are at default:
    eta (learning_rate): 0.3
    max_depth = 6
    min_child_weight = 1
    gamma = 0
    subsample = 1
    lambda = 1 (L2 regularization term on weights. Go up to make more conserv.)
    alpha = 0 (L1 regularization term on weights. Go up to make more conserv.)
    tree_method = auto
    colsample_bytree = 1

"""
#%%
# starting with 300 estimators to make a 1st plot, will keep all else at default.

model = XGBClassifier(n_estimators = 300)

eval_set = [(X_train, y_train), (X_test, y_test)]
eval_metric = ["merror","rmse", "mlogloss", "auc" ]

model.fit(X_train, y_train,
          eval_metric=eval_metric, 
          eval_set=eval_set, 
          verbose=True)

train_merror = model.evals_result()['validation_0']['merror']
test_merror = model.evals_result()['validation_1']['merror']
merror_df = pd.DataFrame({'train': train_merror, 
                          'test': test_merror,
                          'iteration': range(300)})

y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0)) 
#%%
"""
So after 300 rounds the accuracy has gone up by ~1% compared to the default. 
Plotting the eval metrics to see how we're doing.
"""
#%%
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(5,5))
plt.plot( 'iteration', 'train', data=merror_df, marker='', color='olive', linewidth=2, label="train")
plt.plot( 'iteration', 'test', data=merror_df, marker='', color='olive', linewidth=2, linestyle='dashed', label="test")
plt.ylabel('Multiclass classification error rate')
plt.xlabel('Number of estimators')
plt.legend()
plt.show()
#%%
"""
Looks like we max out at about 100 estimators, given what the other params are
set to. For the rest of the parameter tuning, I'll try setting an early stopping
threshold so that I can see if I've run enough estimators. If early stopping
doesn't kick in, then I know I could have added more.


I see learning rate is often set at much lower values than the 0.3 default. 
Tried 0.1, accuracy stalled at 79.7% after either 300 or 600 rounds, but early
stopping didn't kick in for some reason (the training set kept getting better).

Tried 0.01, and accuracy dropped to 76.58%, so 0.1 was better.
Tried learning_rate = 0.2, and colsample_bytree = 0.6 to see if selecting only 
60% of the columns for each tree would help make the model more robust, and not
relying on a few key features. Accuracy went up to 79.83%!

What about longer, but with learning rate at 0.15? Accuracy goes to 79.9%. 
"""
#%%
model = XGBClassifier(n_estimators = 500, 
                      learning_rate = 0.15, 
                      colsample_bytree = 0.6,
                      early_stopping_rounds = 15
                      )

eval_set = [(X_train, y_train), (X_test, y_test)]
eval_metric = ["merror", "mlogloss"]

model.fit(X_train, y_train,
          eval_metric=eval_metric, 
          eval_set=eval_set, 
          verbose=True)

train_merror = model.evals_result()['validation_0']['merror']
test_merror = model.evals_result()['validation_1']['merror']
merror_df = pd.DataFrame({'train': train_merror, 
                          'test': test_merror,
                          'iteration': range(500)})

y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0)) 
#%%
"""
Plot below shows that we really don't get any improvement in accuracy on the test
set with more than 100 trees. It's a lot faster to train if we stick with 100, 
so let's see if we can tune it further sticking with 100.
"""
#%%
fig = plt.figure(figsize=(5,5))
plt.plot( 'iteration', 'train', data=merror_df, marker='', color='olive', linewidth=2, label="train")
plt.plot( 'iteration', 'test', data=merror_df, marker='', color='olive', linewidth=2, linestyle='dashed', label="test")
plt.ylabel('Multiclass classification error rate')
plt.xlabel('Number of estimators')
plt.legend()
plt.show()
#%%
"""
Trying out changes to subsample and max_depth (reducing to use shallower trees). 
With these changes, the actual accuracy doesn't change at 150 estimators, but
when you plot the merror, you see that it now levels out after 80 estimators.
With just 80 trees, I get 79.34% accuracy, and the model should be less overfit.
Changing the gamma (tried 10 and 5), didn't help at all. 
"""
#%%
model = XGBClassifier(n_estimators = 80, 
                      learning_rate = 0.15, 
                      colsample_bytree = 0.6,
                      subsample = 0.7,
                      max_depth = 4,
                      early_stopping_rounds = 15
                      )

eval_set = [(X_train, y_train), (X_test, y_test)]
eval_metric = ["merror", "mlogloss"]

model.fit(X_train, y_train,
          eval_metric=eval_metric, 
          eval_set=eval_set, 
          verbose=True)



y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0)) 
#%%
"""
Conclusion: Tuning the model has reduced the amount of overfitting by using the 
colsample_bytree and subsample parameters, but hasn't made any major improvements
in the overall accuracy. Here are the final graphs for the last model. I really
wanted to get the accuracy up over 80%, but I think this will have to do.

A confusion matrix shows that the model is much worse at distinguishing the bottom
two categories (1 & 2), and that is what's keeping the accuracy low. 
"""
#%%
train_merror = model.evals_result()['validation_0']['merror']
test_merror = model.evals_result()['validation_1']['merror']
merror_df = pd.DataFrame({'train': train_merror, 
                          'test': test_merror,
                          'iteration': range(80)})
    
    
fig = plt.figure(figsize=(5,5))
plt.plot( 'iteration', 'train', data=merror_df, marker='', color='olive', linewidth=2, label="train")
plt.plot( 'iteration', 'test', data=merror_df, marker='', color='olive', linewidth=2, linestyle='dashed', label="test")
plt.ylabel('Multiclass classification error rate')
plt.xlabel('Number of estimators')
plt.legend()
plt.show()

# mlogloss
train_mlogloss = model.evals_result()['validation_0']['mlogloss']
test_mlogloss = model.evals_result()['validation_1']['mlogloss']
mlogloss_df = pd.DataFrame({'train': train_mlogloss, 
                          'test': test_mlogloss,
                          'iteration': range(80)})
    
    
fig = plt.figure(figsize=(5,5))
plt.plot( 'iteration', 'train', data=mlogloss_df, marker='', color='olive', linewidth=2, label="train")
plt.plot( 'iteration', 'test', data=mlogloss_df, marker='', color='olive', linewidth=2, linestyle='dashed', label="test")
plt.ylabel('Multiclass classification log loss')
plt.xlabel('Number of estimators')
plt.legend()
plt.show()

#%% 
"""
A confusion matrix shows that the model is much worse at distinguishing the bottom
two categories (1 & 2), and that is likely what's keeping the accuracy low. 
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
Let's plot the feature importances.
"""
#%%
fig, ax = plt.subplots(figsize=(4, 10))
xgb.plot_importance(model, ax=ax)
plt.show()

#%% 
"""
Testing one last thing - what about balancing the classes?
"""
#%%

g = combined_data.groupby('source')
balanced_data = pd.DataFrame(g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True)))


X = balanced_data.drop(["text", "source"], axis = 1)
Y = balanced_data['source'] # "source" is the column of numeric sources

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

# train the model
model = XGBClassifier(n_estimators = 80, 
                      learning_rate = 0.15, 
                      colsample_bytree = 0.6,
                      subsample = 0.7,
                      max_depth = 4,
                      early_stopping_rounds = 15
                      )

eval_set = [(X_train, y_train), (X_test, y_test)]
eval_metric = ["merror", "mlogloss"]

model.fit(X_train, y_train,
          eval_metric=eval_metric, 
          eval_set=eval_set, 
          verbose=True)



y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0)) 

plot_confusion_matrix(y_test, y_pred, classes = y_test, normalize=True,
                      title='Normalized confusion matrix')


filename = './flask_app/canisaythat_aws/model/test_XGBoost_model.sav'
pickle.dump(model, open(filename, 'wb'))

filename = './flask_app/canisaythat_aws/model/test_XGBoost_scaler.sav'
pickle.dump(scaler, open(filename, 'wb'))