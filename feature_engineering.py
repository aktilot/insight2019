#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%%
import pandas as pd
import numpy as np
all_data = pd.read_csv("./data/190617_corpus.csv", index_col = 0)
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

#top_two_comp = pca_df[['pca1','pca2']] # taking first and second principal component
#
#textstat_results["source"] = textstat_results["source"].astype('category')
#textstat_results.dtypes
#textstat_results["source_recode"] = textstat_results["source"].cat.codes
#
#textstat_scatter(top_two_comp.values, textstat_results["source_recode"])

#Categories (5, object): 
#[Dissertation, 
#Extremely Casual, 
#Governmental, 
#Slack-like, 
#Workplace_Casual]

#%%
"""
Trying TSNE to see the clusters. This takes a few minutes.
"""
#%%
#from sklearn.manifold import TSNE
#
#textstat_tsne = TSNE(random_state=42).fit_transform(textstat_results)
#
#textstat_scatter(textstat_tsne, textstat_results["source"])

#%%
"""
Let's see what kind of classification we can do, will try XGBoost first.
64% after cleaning up the corpus. We can do better! Let's add more!
"""
#%%
#from xgboost import XGBClassifier
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score
## split data into X and y
#X = textstat_results.iloc[:,0:8]
#Y = textstat_results.iloc[:,10]
## split data into train and test sets
#seed = 7
#test_size = 0.33
#X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
## fit model no training data
#model = XGBClassifier()
#model.fit(X_train, y_train)
## make predictions for test data
#y_pred = model.predict(X_test)
#predictions = [round(value) for value in y_pred]
## evaluate predictions
#accuracy = accuracy_score(y_test, predictions)
#print("Accuracy: %.2f%%" % (accuracy * 100.0))
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
print("Accuracy: %.2f%%" % (accuracy * 100.0)) #71.79% with 190617 corpus
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
Is r/AskHistorians throwing off the Slack-like section? Let's look!
"""
#%%
no_AskHistorians = combined_data.loc[combined_data["subreddit"] != "AskHistorians",:]

# split data into X and y
X = no_AskHistorians.iloc[:,3:14]
Y = no_AskHistorians['source'] # "source" is the column of numeric sources

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
print("Accuracy: %.2f%%" % (accuracy * 100.0)) #74.06% without AskHistorians
#%%
"""
Now we know we need to work on classifying "Slack-like" away from "Extremely Casual"
What about consecutive punctuation?
"""
#%%
import re
def scream_counter(text):
    crazy_confused = len(re.findall(r'!\?|\?!', text))
    ahhhh = len(re.findall(r'!!', text))
    huhhh = len(re.findall(r'\?\?', text))
    screams = crazy_confused + ahhhh + huhhh            
    return screams
    
    
internet_yelling = []
for row in combined_data["text"]:
    screams = scream_counter(str(row))
    internet_yelling.append(screams)

combined_data["Yell_count"] = internet_yelling
combined_data["Yell_count"].value_counts()
#%%
"""
Let's see if adding yells helped.
"""
#%%
no_AskHistorians = combined_data.loc[combined_data["subreddit"] != "AskHistorians",:]
no_AskHistorians = no_AskHistorians.reset_index(drop=True) #Maybe this is why y_text was showing NaN?

# split data into X and y
X = no_AskHistorians.iloc[:,3:15]
Y = no_AskHistorians['source'] # "source" is the column of numeric sources

# split data into train and test sets
seed = 87
test_size = 0.2
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
print("Accuracy: %.2f%%" % (accuracy * 100.0)) #73.84%, so no real change?
#%%
"""
It did not. What about adding or swapping that summary grade level from textstat?
"""
#%%

text_standard = []
for i in combined_data["text"]:
    grade = ts.text_standard(str(i), float_output=True)
    text_standard.append(grade)

combined_data["text_standard"] = text_standard
sns.boxplot( x=combined_data["source"], y=combined_data["text_standard"] )

#%%
"""
Sanity check: using LIME see why the model is doing well so far
Tutorial: https://www.analyticsvidhya.com/blog/2017/06/building-trust-in-machine-learning-models/
"""
#%%
import lime
import lime.lime_tabular

# create lambda function to return probability of the target variable given a set of features
predict_fn_xgb = lambda x: model.predict_proba(x).astype(float)
#create list of feature names to be used later
feature_names = no_AskHistorians.columns[3:15].tolist()
#create LIME explainer
explainer = lime.lime_tabular.LimeTabularExplainer(X_train, 
                                                   feature_names = feature_names, 
                                                   class_names = ['1', '2','3','4','5'],
                                                   kernel_width = 3)
#%%
"""
Sanity check: test specific items to see why they were labeled
"""
#%%
lime_labled_tuples = list(zip(y_test, predictions))
lime_labeled_df = pd.DataFrame(lime_labled_tuples, columns = ["true_score", "predicted_score"])

observation_to_check = 80 #used Variable explorer to figure this out
exp = explainer.explain_instance(X_test[observation_to_check], 
                                 predict_fn_xgb, 
                                 num_features = len(feature_names))

import pickle
filename = 'pickled_LIME.sav'
pickle.dump(exp, open(filename, 'wb'))


weirdos = [30,80,99,322,380, 446]

#%%
"""
Sanity check: What if we remove the URLs? Hopefully this won't change the accuracy.
Let's use just the standard scaler and the engineered features and see what we get. 
Yikes, it was down to 60.7%, that's no good. Add all features back and we're at #74.36%! 
"""
#%%
# remove hyperlinks
no_AskHistorians["text"] = no_AskHistorians["text"].str.replace(r'http\S*\s', ' ')
no_AskHistorians["text"] = no_AskHistorians["text"].str.replace(r'http\S*(\n|\)|$)', ' ')

textstat_results2 = pd.DataFrame(columns = ['flesch_ease', 'flesch_grade','gfog',
           'auto_readability','cl_index','lw_formula','dcr_score', 'syll_count', 'lex_count'])

for i in no_AskHistorians["text"]: #textstat needs a string
    results = textstat_stats(str(i))
    textstat_results2 = textstat_results2.append(results, ignore_index=True) #so that index is continuous

textstat_results2 = textstat_results2.reset_index(drop=True)

text_standard = []
for i in no_AskHistorians["text"]:
    grade = ts.text_standard(str(i), float_output=True)
    text_standard.append(grade)

no_AskHistorians_noURLs = no_AskHistorians.iloc[:,0:3]
no_AskHistorians_noURLs = pd.concat([no_AskHistorians_noURLs, textstat_results2], axis = 1)
no_AskHistorians_noURLs["text_standard"] = text_standard

internet_yelling = []
for row in no_AskHistorians_noURLs["text"]:
    screams = scream_counter(str(row))
    internet_yelling.append(screams)

no_AskHistorians_noURLs["Yell_count"] = internet_yelling

any_bad = []
for row in no_AskHistorians_noURLs["text"]:
    if any(str(word) in str(row) for word in bad_words):
        any_bad.append(1)
    else: any_bad.append(0)

no_AskHistorians_noURLs["Google_curses"] = any_bad

emoji_counts = []
for row in no_AskHistorians_noURLs["text"]:
    emoji_num = len(emoji_counter(str(row)))
    emoji_counts.append(emoji_num)

no_AskHistorians_noURLs["Num_emoji"] = emoji_counts


# split data into X and y
X = no_AskHistorians_noURLs.iloc[:,3:16]
Y = no_AskHistorians_noURLs['source'] # "source" is the column of numeric sources

# split data into train and test sets
seed = 8
test_size = 0.2
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
print("Accuracy: %.2f%%" % (accuracy * 100.0)) #74.36%! It's actually a smidge better!

#%%
"""
Ok, so we get up to 74% accuracy with essentially just readability statistics. What about
other features that are more actionable for the user?
Actionable features: 
    - Foreign words
    - Parts of speech?
    - Sentence length
    - strength of the sentiment

"""
#%%
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA

sia = SIA()
results = []

for line in no_AskHistorians_noURLs["text"]: # this returns a list of dicts
    pol_score = sia.polarity_scores(line)
    pol_score['text'] = line
    results.append(pol_score)

sia_neg = []
sia_pos = []
sia_neu = []
sia_comp = []
for document in results:
    neg = document['neg']
    pos = document['pos']
    neu = document['neu']
    comp = document['compound']
    sia_neg.append(neg)
    sia_pos.append(pos)
    sia_neu.append(neu)
    sia_comp.append(comp)
    
no_AskHistorians_noURLs["SIA_neg"] = sia_neg
no_AskHistorians_noURLs["SIA_pos"] = sia_pos
no_AskHistorians_noURLs["SIA_neu"] = sia_neu
no_AskHistorians_noURLs["SIA_com"] = sia_comp

# split data into X and y
X = no_AskHistorians_noURLs.iloc[:,3:20]
Y = no_AskHistorians_noURLs['source'] # "source" is the column of numeric sources

# split data into train and test sets
seed = 8
test_size = 0.2
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

# Plot normalized confusion matrix
plot_confusion_matrix(y_test, y_pred, classes = y_test, normalize=True,
                      title='Normalized confusion matrix')

#%%
"""
Density plot with labels for each data source, to see how I'm doing.
Code adapted from: https://towardsdatascience.com/histograms-and-density-plots-in-python-f6bda88f5ac0
"""
#%%
sources = pd.unique(no_AskHistorians_noURLs["source"])

# Positive sentiment
for source in sources:
    # Subset to the airline
    subset = no_AskHistorians_noURLs[no_AskHistorians_noURLs['source'] == source]
    # Draw the density plot
    sns.distplot(subset['SIA_pos'], hist = False, kde = True,
                 kde_kws = {'linewidth': 3},
                 label = source)
    
# Plot formatting
plt.legend(prop = {'size': 8}, title = 'Data source', loc = 'best')
plt.title('Density Plot with Multiple Text Sources')
plt.xlabel('Sentiment Intensity: Positive')
plt.ylabel('Density')

# Negative sentiment
for source in sources:
    # Subset to the airline
    subset = no_AskHistorians_noURLs[no_AskHistorians_noURLs['source'] == source]
    # Draw the density plot
    sns.distplot(subset['SIA_neg'], hist = False, kde = True,
                 kde_kws = {'linewidth': 3},
                 label = source)
    
# Plot formatting
plt.legend(prop = {'size': 8}, title = 'Data source', loc = 'best')
plt.title('Density Plot with Multiple Text Sources')
plt.xlabel('Sentiment Intensity: Negative')
plt.ylabel('Density')

# Neutral sentiment
for source in sources:
    # Subset to the airline
    subset = no_AskHistorians_noURLs[no_AskHistorians_noURLs['source'] == source]
    # Draw the density plot
    sns.distplot(subset['SIA_neu'], hist = False, kde = True,
                 kde_kws = {'linewidth': 3},
                 label = source)
    
# Plot formatting
plt.legend(prop = {'size': 8}, title = 'Data source', loc = 'best')
plt.title('Density Plot with Multiple Text Sources')
plt.xlabel('Sentiment Intensity: Neutral')
plt.ylabel('Density')

# Compound sentiment
for source in sources:
    # Subset to the airline
    subset = no_AskHistorians_noURLs[no_AskHistorians_noURLs['source'] == source]
    # Draw the density plot
    sns.distplot(subset['SIA_com'], hist = False, kde = True,
                 kde_kws = {'linewidth': 3},
                 label = source)
    
# Plot formatting
plt.legend(prop = {'size': 8}, title = 'Data source', loc = 'best')
plt.title('Density Plot with Multiple Text Sources')
plt.xlabel('Sentiment Intensity: Compound')
plt.ylabel('Density')
