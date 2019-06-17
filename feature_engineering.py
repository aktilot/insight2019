#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%%
import pandas as pd
import numpy as np
all_data = pd.read_csv("./data/190615_corpus.csv", index_col = 0)
#%%
"""
Start with some basics - what's the variance in reading level for these docs?
Thanks SO for the idea on bundling textstat into a function:
    https://stackoverflow.com/questions/45549188/pandas-difficulty-adding-new-columns
"""
#%%
import nltk
import textstat as ts

def textstat_stats(text):
    flesch_ease = ts.flesch_reading_ease(text)
    flesch_grade = ts.flesch_kincaid_grade(text)
    gfog = ts.gunning_fog(text)
    smog = ts.smog_index(text)
    auto_readability = ts.automated_readability_index(text)
    cl_index = ts.coleman_liau_index(text)
    lw_formula = ts.linsear_write_formula(text)
    dcr_score = ts.dale_chall_readability_score(text)
#    idx = ['flesch_ease', 'flesch_grade','gfog',
#           'smog','auto_readability','cl_index','lw_formula','dcr_score']
    return pd.Series([flesch_ease, flesch_grade, gfog, 
                      smog, auto_readability, cl_index, lw_formula, dcr_score])

#%%
"""
Apply textstat functions to corpus and do a little exploration
"""
#%%
textstat_results = pd.DataFrame()

for i in all_data["text"]: #textstat needs a string
    results = textstat_stats(str(i))
    textstat_results = textstat_results.append(results, ignore_index=True) #so that index is continuous

textstat_results["source"] = all_data["source"]
#%%
"""
Cont.
PCA code from DataCamp
"""
#%%
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns

def textstat_scatter(x, colors):
    # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))

    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # add the labels for each digit corresponding to the label
    txts = []

    for i in range(num_classes):

        # Position of each label at median of data points.

        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts

from sklearn.decomposition import PCA


pca = PCA(n_components=4)
pca_result = pca.fit_transform(textstat_results.iloc[:,1:8])

pca_df = pd.DataFrame(columns = ['pca1','pca2','pca3','pca4'])

pca_df['pca1'] = pca_result[:,0]
pca_df['pca2'] = pca_result[:,1]
pca_df['pca3'] = pca_result[:,2]
pca_df['pca4'] = pca_result[:,3]

print('Variance explained per principal component: {}'.format(pca.explained_variance_ratio_))
#%%
"""
Recode the sources into numeric so that I can plot them 
"""
#%%

top_two_comp = pca_df[['pca1','pca2']] # taking first and second principal component

textstat_results["source"] = textstat_results["source"].astype('category')
textstat_results.dtypes
textstat_results["source_recode"] = textstat_results["source"].cat.codes

textstat_scatter(top_two_comp.values, textstat_results["source_recode"])

#Categories (5, object): 
#[Dissertation, 
#Extremely Casual, 
#Governmental, 
#Slack-like, 
#Workplace_Casual]

#%%
## Trying TSNE to see the clusters

#%%
from sklearn.manifold import TSNE

textstat_tsne = TSNE(random_state=42).fit_transform(textstat_results.iloc[:,0:7])

textstat_scatter(textstat_tsne, textstat_results["source_recode"])

#%%
## Let's see what kind of classification we can do
# will try XGBoost first
#%%
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# split data into X and y
X = textstat_results.iloc[:,0:7]
Y = textstat_results.iloc[:,9]
# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))