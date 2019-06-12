#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Re-run model with custom stopwords
"""
#%%
import os
import csv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

#%%
"""
Read in data from prototyping_model.py
"""
#%% 
all_data = pd.read_csv("../data/190611_corpus.csv", index_col = 0)
all_data["text"] = all_data["text"].fillna(' ') #so that the TfidfVectorizer will work


my_stopwords = []
with open('../data/custom_stops_plus_nltk.csv', 'r') as f:
    reader = csv.reader(f)
    my_stopwords = list(reader)
my_stopwords = [item for sublist in my_stopwords for item in sublist]

#%%
"""
Define model and run it.
"""
#%% 

X_train, X_test, y_train, y_test = train_test_split(all_data["text"], all_data["score"], 
                                                    test_size=0.2, 
                                                    random_state=40)

pipeline_tfidf_custom = Pipeline(steps=[('vectorizer', TfidfVectorizer(ngram_range=(1, 2),
                                                                encoding='utf-8',
                                                                decode_error='replace',
                                                                lowercase=False,
                                                                stop_words=my_stopwords)),
                                 ('classifier', LogisticRegression())])
pipeline_tfidf_custom.fit(X_train, y_train)
#print('tfidf', pipeline_tfidf.score(X_test, y_test)) #currently 99.5%
y_pred = pipeline_tfidf_custom.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(pipeline_tfidf_custom.score(X_test, y_test)))
#%%
"""
Calculating probabilities and attaching them to the labels for the test set.
"""
#%%
test_prob = pipeline_tfidf_custom.predict_proba(X_test)[:,1]
test_prob = pd.DataFrame(test_prob, columns = ["Probability"])
y_labels = pd.DataFrame(y_test)
y_labels = y_labels.join(all_data["subreddit"], how='left')
y_labels = y_labels.reset_index()
test_prob_labeled = y_labels.join(test_prob, how="left") 
#%%
"""
Density plot with labels for each data source, to see how I'm doing.
Code adapted from: https://towardsdatascience.com/histograms-and-density-plots-in-python-f6bda88f5ac0
"""
#%%
subreddits = pd.unique(test_prob_labeled["subreddit"])

# Iterate through the five airlines
for subr in subreddits:
    # Subset to the airline
    subset = test_prob_labeled[test_prob_labeled['subreddit'] == subr]
    
    # Draw the density plot
    sns.distplot(subset['Probability'], hist = False, kde = True,
                 kde_kws = {'linewidth': 3},
                 label = subr)
    
# Plot formatting
plt.legend(prop = {'size': 8}, title = 'Data source', loc = 'best')
plt.title('Density Plot with Multiple Text Sources')
plt.xlabel('Probability of being labeled "Professional"')
plt.ylabel('Density')