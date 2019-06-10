#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
import ijson
import seaborn as sns
#%%
"""
Here we're loading in the datasets and munging
"""
#%%
# hp #32,562 comments
hp = pd.read_csv("../data/entertainment_harrypotter.csv", header = None)
hp = hp.drop([0,1], axis = 1)
hp.columns = ["text","id","subreddit","meta","time","author","ups","downs","authorlinkkarma","authorkarma","authorisgold"]
hp = hp.dropna(subset = ["text"])
hp["score"] = 0
hp["tone"] = "unprofessional"

# ah # 8,835 comments
ah = pd.read_csv("../data/learning_askhistorians.csv", header = None)
ah = ah.drop([0], axis = 1)
ah.columns = ["text","id","subreddit","meta","time","author","ups","downs","authorlinkkarma","authorkarma","authorisgold"]
ah = ah.dropna(subset = ["text"])
ah["score"] = 1
ah["tone"] = "professional"

# first 8,000 rows of the enron dataset
enron = pd.read_csv("../data/enron_05_17_2015_with_labels_v2.csv", nrows = 8000) #517,401 emails, some are labeled with
enron.rename(columns={'content': 'text'}, inplace=True)
enron["score"] = 1
enron["tone"] = "professional"
enron["subreddit"] = "enron"
#%%
"""
Split the training (80%) and test (20%) data Building our model pipeline
"""
#%%
all_data = pd.concat([ah.loc[1:8000,["text","subreddit","tone", "score"]], 
                      hp.loc[1:8000,["text","subreddit","tone", "score"]],
                      enron.loc[:,["text","subreddit","tone", "score"]]],
                    sort = False)
all_data["text"] = all_data["text"].fillna(' ') #so that the TfidfVectorizer will work

list_corpus = all_data["text"].tolist()
list_labels = all_data["score"].tolist()

X_train, X_test, y_train, y_test = train_test_split(list_corpus, list_labels, 
                                                    test_size=0.2, 
                                                    random_state=40)
#%%
"""
Building our model pipeline
"""
#%%
pipeline_tfidf = Pipeline(steps=[('vectorizer', TfidfVectorizer(ngram_range=(1, 2),
                                                                encoding='utf-8',
                                                                decode_error='replace',
                                                                lowercase=False,
                                                                stop_words='english')),
                                 ('classifier', LogisticRegression())])
#%%
"""
Training the model on the X_train set of 17,900 rows
Testing it on the X_test set of 4,477 rows
"""
#%%
pipeline_tfidf.fit(X_train, y_train)
#print('tfidf', pipeline_tfidf.score(X_test, y_test)) #currently 99.5%
y_pred = pipeline_tfidf.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(pipeline_tfidf.score(X_test, y_test)))
#%%

#%%
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
#%%

#%%
train_prob = pipeline_tfidf.predict_proba(X_test)[:,1]
train_prob_labeled = pd.concat([train_prob, ])

sns.distplot(train_prob, hist = False, kde = True,
                 kde_kws = {'linewidth': 3})
#%%
