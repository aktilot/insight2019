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
    flesch_ease = ts.flesch_reading_ease(text) #Flesch Reading Ease Score
    flesch_grade = ts.flesch_kincaid_grade(text) #Flesch-Kincaid Grade Level
    gfog = ts.gunning_fog(text) # FOG index, also indicates grade level
    #smog = ts.smog_index(text) # SMOG index, also indicates grade level, only useful on 30+ sentences
    auto_readability = ts.automated_readability_index(text) #approximates the grade level needed to comprehend the text.
    cl_index = ts.coleman_liau_index(text) #grade level of the text using the Coleman-Liau Formula.
    lw_formula = ts.linsear_write_formula(text) #grade level using the Linsear Write Formula.
    dcr_score = ts.dale_chall_readability_score(text) #uses a lookup table of the most commonly used 3000 English words
    # textstat.text_standard(text, float_output=False) # summary of all the grade level functions
    syll_count = ts.syllable_count(text, lang='en_US')
    lex_count = ts.lexicon_count(text, removepunct=True)
    idx = ['flesch_ease', 'flesch_grade','gfog',
           'auto_readability','cl_index','lw_formula','dcr_score', 'syll_count', 'lex_count']
    return pd.Series([flesch_ease, flesch_grade, gfog, 
                      auto_readability, cl_index, lw_formula, dcr_score, syll_count, lex_count], index = idx)

#%%
"""
Apply textstat functions to corpus. This takes ~2 minutes.
"""
#%%
textstat_results = pd.DataFrame(columns = ['flesch_ease', 'flesch_grade','gfog',
           'auto_readability','cl_index','lw_formula','dcr_score', 'syll_count', 'lex_count'])

for i in all_data["text"]: #textstat needs a string
    results = textstat_stats(str(i))
    textstat_results = textstat_results.append(results, ignore_index=True) #so that index is continuous

textstat_results = textstat_results.reset_index(drop=True)

## Rerun these steps below if I need to reload all_data
all_data = all_data.reset_index(drop=True)
combined_data = pd.concat([all_data, textstat_results], axis = 1)

sources_dict = {'Extremely Casual': 1, 'Slack-like': 2, 'Workplace_Casual': 3, 
                'Governmental': 4, 'Dissertation': 5}

combined_data.replace(sources_dict, inplace=True)


#textstat_results["source"] = all_data["source"]
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
pca_result = pca.fit_transform(textstat_results.iloc[:,0:8])

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
"""
Trying TSNE to see the clusters
"""
#%%
from sklearn.manifold import TSNE

textstat_tsne = TSNE(random_state=42).fit_transform(textstat_results.iloc[:,0:7])

textstat_scatter(textstat_tsne, textstat_results["source_recode"])

#%%
"""
Let's see what kind of classification we can do, will try XGBoost first.
64% after cleaning up the corpus. We can do better! Let's add more!
"""
#%%
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# split data into X and y
X = textstat_results.iloc[:,0:8]
Y = textstat_results.iloc[:,10]
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
#%%
"""
Let's see what kind of classification we can do, will try XGBoost first.
Answer: 64% after cleaning up the corpus. We can do better! Let's add more!

Starting with curse words.
"""
#%%
#urban_df = pd.read_csv("./data/urban_dictionary_cleaned.csv", index_col = 0)
#
#bad_words = urban_df["word"].tolist()
#
#any_bad = []
#for row in all_data["text"]:
#    if any(str(word) in str(row) for word in bad_words):
#        any_bad.append(1)
#    else: any_bad.append(0)
#
#all_data["urban_dict_yn"] = any_bad
#all_data["urban_dict_yn"].value_counts() # yikes, this is not informative


Google_Curses = pd.read_csv("./data/RobertJGabriel_Google_swear_words.txt", header = None)
bad_words = Google_Curses[0].tolist()

any_bad = []
for row in combined_data["text"]:
    if any(str(word) in str(row) for word in bad_words):
        any_bad.append(1)
    else: any_bad.append(0)

combined_data["Google_curses"] = any_bad
combined_data["Google_curses"].value_counts() #much better, only 5817 with a curse
#%%
"""
Now we'll add emoji using the emoji package
Thanks: https://stackoverflow.com/questions/43146528/how-to-extract-all-the-emojis-from-text
"""
#%%
import emoji
import regex

def emoji_counter(text):
    emoji_list = []
    data = regex.findall(r'\X', text)
    for word in data:
        if any(char in emoji.UNICODE_EMOJI for char in word):
            emoji_list.append(word)
    return emoji_list

emoji_counts = []
for row in combined_data["text"]:
    emoji_num = len(emoji_counter(str(row)))
    emoji_counts.append(emoji_num)

combined_data["Num_emoji"] = emoji_counts
combined_data["Num_emoji"].value_counts()
#%%
"""
Let's check this again, and include a confusion matrix on the XGBoost result.
"""
#%%
from sklearn import preprocessing

combined_data = combined_data.dropna(subset=['source'], axis = 0) #remove 1 weird row
combined_data["source"].value_counts()

# split data into X and y
X = combined_data.iloc[:,3:14]
Y = combined_data['source'] # "source" is the column of numeric sources

# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

# This time we will scale the data correctly
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test) 

# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
#%%
"""
The confusion matrxix
Code for plots from: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
"""
#%%
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

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

# Plot non-normalized confusion matrix
plot_confusion_matrix(y_test, y_pred, classes = y_test,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(y_test, y_pred, classes = y_test, normalize=True,
                      title='Normalized confusion matrix')

plt.show()
#%%
"""
More features! Let's try punctuation usage
"""
#%%