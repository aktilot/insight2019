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
import matplotlib.pyplot as plt
#%%
"""
Here we're loading in the datasets and labeling them.
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
Clean the data.
"""
#%%
all_data = pd.concat([ah.loc[1:8000,["text","subreddit","tone", "score"]], 
                      hp.loc[1:8000,["text","subreddit","tone", "score"]],
                      enron.loc[:,["text","subreddit","tone", "score"]]],
                    sort=False, 
                    ignore_index=True)
all_data["text"] = all_data["text"].fillna(' ') #so that the TfidfVectorizer will work

# function to remove URLs from the data. May add more bits.
# adapted from https://github.com/hundredblocks/concrete_NLP_tutorial/blob/master/NLP_notebook.ipynb
def standardize_text(df, text_field):
    df[text_field] = df[text_field].str.replace(r"http\S+", "")
    df[text_field] = df[text_field].str.replace(r"http", "")
    return df

all_data["text"] = standardize_text(all_data, "text")


#%%
"""
Split the training (80%) and test (20%) data Building our model pipeline
"""
#%%
#list_corpus = all_data["text"].tolist()
#list_labels = all_data["score"].tolist()

X_train, X_test, y_train, y_test = train_test_split(all_data["text"], all_data["score"], 
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
"""
Calculating probabilities and attaching them to the labels for the test set.
"""
#%%
test_prob = pipeline_tfidf.predict_proba(X_test)[:,1]
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
plt.legend(prop={'size': 16}, title = 'Data source')
plt.title('Density Plot with Multiple Text Sources')
plt.xlabel('Probability of being labeled "Professional"')
plt.ylabel('Density')
#%%
