#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 10:05:43 2019

@author: amandakomuro
"""

#%%
import os
import csv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import utils
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
import pickle


#%%
"""
Read in data from prototyping_model.py
"""
#%% 
all_data = pd.read_csv("./data/190613_corpus.csv", index_col = 0)
all_data["text"] = all_data["text"].fillna(' ') #so that the TfidfVectorizer will work


my_stopwords = []
with open('./data/custom_stops_plus_nltk.csv', 'r') as f:
    reader = csv.reader(f)
    my_stopwords = list(reader)
my_stopwords = [item for sublist in my_stopwords for item in sublist]

#%%
"""
Split and tokenize the text.
doc2vec pipeline from: https://medium.com/@mishra.thedeepak/doc2vec-simple-implementation-example-df2afbbfbad5
Some texts end up with no words left after removing stopwords. Not sure if I should clean those out.
"""
#%% 
#X_train, X_test, y_train, y_test = train_test_split(all_data["text"], all_data["score"], 
#                                                    test_size=0.2, 
#                                                    random_state=40)

train, test = train_test_split(all_data, test_size=0.2, random_state=42)

def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text): #sent is for sentence
        for word in nltk.word_tokenize(sent, language='english'): # language='english'
            if word in my_stopwords: #using my custom set of subreddit stopwords plus NLTK's
                continue
            tokens.append(word) #(word.lower()) if I want to make everything lowercase.
    return tokens

train_tagged = train.apply(
    lambda r: TaggedDocument(words=tokenize_text(r['text']), tags=[r.tone]), axis=1)
test_tagged = test.apply(
    lambda r: TaggedDocument(words=tokenize_text(r['text']), tags=[r.tone]), axis=1)

#%%
"""
doc2vec pipeline from: https://medium.com/@mishra.thedeepak/doc2vec-simple-implementation-example-df2afbbfbad5
"""
#%% 
import multiprocessing
cores = multiprocessing.cpu_count()

model_dbow = Doc2Vec(dm=0, 
                     vector_size=300, 
                     negative=5, 
                     hs=0, 
                     min_count=2, 
                     sample = 0, 
                     workers=cores)

model_dbow.build_vocab([x for x in tqdm(train_tagged.values)])
#%%
"""
Building a Vocabulary
"""
#%% 
for epoch in range(30):
    model_dbow.train(utils.shuffle([x for x in tqdm(train_tagged.values)]), total_examples=len(train_tagged.values), epochs=1)
    model_dbow.alpha -= 0.002
    model_dbow.min_alpha = model_dbow.alpha
#%% 
"""
Building the Final Vector Feature for the Classifier
"""
#%% 
def vec_for_learning(model, tagged_docs):
    sents = tagged_docs.values
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, regressors
#%% 
"""
Train the Logistic Regression Classifier.
"""
#%%
y_train, X_train = vec_for_learning(model_dbow, train_tagged)
y_test, X_test = vec_for_learning(model_dbow, test_tagged)

logreg = LogisticRegression(n_jobs=1, C=1e5)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

from sklearn.metrics import accuracy_score, f1_score

print('Testing accuracy %s' % accuracy_score(y_test, y_pred))
print('Testing F1 score: {}'.format(f1_score(y_test, y_pred, average='weighted')))
#%% 
"""
Generate probabilities of being labeled "professional" or "unprofessional"
"""
#%%
test_prob = logreg.predict_proba(X_test)
test_prob = pd.DataFrame(test_prob[:,[0]], columns = ["Probability"])
y_labels = test.reset_index()
test_prob_labeled = test_prob.join(y_labels.loc[:,["subreddit", "text"]], how="left") # so I can check sensibility
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
#%%
"""
"Ultimate test" student emails hand-labeled as "professional"
"""
#%%
# read data
with open("./data/ultimate_test_prof.csv") as f:
  ut_prof = f.readlines()
  
# add tags
ut_prof = pd.DataFrame(ut_prof,columns=['text'])
ut_prof["tone"] = "professional"

# tokenize
ut_tagged = ut_prof.apply(
    lambda r: TaggedDocument(words=tokenize_text(r['text']), tags=[r.tone]), axis=1)

# create vectorized version
y_ut_prof, X_ut_prof = vec_for_learning(model_dbow, ut_tagged)

# predict professionalism using trained model from above
ut_prof_prob = logreg.predict_proba(X_ut_prof)

ut_prof_prob
#%%
"""
"Ultimate test" student emails hand-labeled as "unprofessional"
"""
#%%
# read data
with open("./data/ultimate_test_unprof.csv") as f:
  ut_prof = f.readlines()
  
# add tags
ut_unprof = pd.DataFrame(ut_prof,columns=['text'])
ut_unprof["tone"] = "unprofessional"

# tokenize
ut_unprof_tagged = ut_unprof.apply(
    lambda r: TaggedDocument(words=tokenize_text(r['text']), tags=[r.tone]), axis=1)

# create vectorized version
y_ut_unprof, X_ut_unprof = vec_for_learning(model_dbow, ut_unprof_tagged)

# predict professionalism using trained model from above
ut_unprof_prob = logreg.predict_proba(X_ut_unprof)

ut_unprof_prob
#%%
"""
Pickle the model and needed objects for web app.
"""
#%%
# model_dbow.save('./insight2019/flask_app/my_flask/model/canisaythat_d2v.model')
filename = './insight2019/flask_app/my_flask/model/finalized_model.sav'
pickle.dump(model_dbow, open(filename, 'wb'))

filename2 = './insight2019/flask_app/my_flask/model/finalized_model2.sav'
pickle.dump(logreg, open(filename2, 'wb'))