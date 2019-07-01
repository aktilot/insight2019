#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
#%%
"""
Custom functions for feature engineering
"""
#%%
import nltk
import textstat as ts
import emoji
import regex
import re

def textstat_stats(text):
    doc_length = len(text.split()) 
    flesch_ease = ts.flesch_reading_ease(text) #Flesch Reading Ease Score
    flesch_grade = ts.flesch_kincaid_grade(text) #Flesch-Kincaid Grade Level
    gfog = ts.gunning_fog(text) # FOG index, also indicates grade level
#    smog = ts.smog_index(text) # SMOG index, also indicates grade level, only useful on 30+ sentences
    auto_readability = ts.automated_readability_index(text) #approximates the grade level needed to comprehend the text.
    cl_index = ts.coleman_liau_index(text) #grade level of the text using the Coleman-Liau Formula.
    lw_formula = ts.linsear_write_formula(text) #grade level using the Linsear Write Formula.
    dcr_score = ts.dale_chall_readability_score(text) #uses a lookup table of the most commonly used 3000 English words
#    text_standard = ts.text_standard(text, float_output=False) # summary of all the grade level functions
    syll_count = ts.syllable_count(text, lang='en_US')
    syll_count_scaled = syll_count / doc_length
    lex_count = ts.lexicon_count(text, removepunct=True)
    lex_count_scaled = lex_count / doc_length
    idx = ['flesch_ease', 'flesch_grade','gfog',
           'auto_readability','cl_index','lw_formula',
           'dcr_score', 
#           'text_standard', 
           'syll_count', 'lex_count']
    return pd.Series([flesch_ease, flesch_grade, gfog, 
                      auto_readability, cl_index, lw_formula, 
                      dcr_score, 
#                      text_standard, 
                      syll_count_scaled, lex_count_scaled], index = idx)

def emoji_counter(text):
    emoji_list = []
    data = regex.findall(r'\X', text)
    for word in data:
        if any(char in emoji.UNICODE_EMOJI for char in word):
            emoji_list.append(word)
    return emoji_list

def scream_counter(text):
    crazy_confused = len(re.findall(r'!\?|\?!', text))
    ahhhh = len(re.findall(r'!!', text))
    huhhh = len(re.findall(r'\?\?', text))
    screams = crazy_confused + ahhhh + huhhh            
    return screams


Google_Curses = pd.read_csv("./data/custom_curse_words.txt", header = None)
bad_words = Google_Curses[0].tolist()

def swear_counter(text): #returns number of curses in text
    swear_jar = []
    what_curses = []
    for word in bad_words:
        curses = text.count(word)
        swear_jar.append(curses)
        if curses > 0:
            what_curses.append(word)
    return sum(swear_jar), what_curses

contractions = [
  r'n\'t',
  r'I\'m', 
  r'(\w+)\'ll', 
  r'(\w+)n\'t', 
  r'(\w+)\'ve', 
  r'(\w+)\'s', 
  r'(\w+)\'re', 
  r'(\w+)\'d',
]

def contraction_counter(text):
    doc_length = len(text.split()) 
    abbreviations = []
    for abbrev in contractions:
        num_abbrevs = len(re.findall(abbrev, text))
        abbreviations.append(num_abbrevs)
    return sum(abbreviations) / doc_length
    
    
#%%
"""
Final cleaning of the final corpus (190617):
    - remove URLs
    - remove r/AskHistorians
    - recode source into numeric
    - drop any rows missing a source
"""
#%% 
# remove AskHistorians and reset index (this is important)
all_data = pd.read_csv("./data/190617_corpus.csv", index_col = 0)
clean_data = all_data.loc[all_data["subreddit"] != "AskHistorians",:]
clean_data = clean_data.reset_index(drop=True) 

# remove hyperlinks
clean_data["text"] = clean_data["text"].str.replace(r'http\S*\s', ' ')
clean_data["text"] = clean_data["text"].str.replace(r'http\S*(\n|\)|$)', ' ')
clean_data["text"] = clean_data["text"].str.replace(r'', ' ')
          
        
# recode the sources (needed for model?)
sources_dict = {'Extremely Casual': 1, 'Slack-like': 2, 'Workplace_Casual': 3, 
                'Governmental': 4, 'Dissertation': 5}
clean_data.replace(sources_dict, inplace=True)        

# removing rows without a source
clean_data = clean_data.dropna(subset=['source'], axis = 0) #remove 1 weird row
clean_data["source"].value_counts()

# maybe also remove rows that end up with nothing left in the text?
doc_lengths = [len(text.strip().split()) for text in clean_data['text']]
clean_data["doc_length"] = doc_lengths
clean_data = clean_data[clean_data["doc_length"] != 0] # removes about 40 rows
clean_data = clean_data.drop(["doc_length"], axis=1) 
clean_data = clean_data.reset_index(drop=True) # so we must reset the index
#%%
"""
First create features from NLP packages:
    - textstat for readability measures (they're not all the same)
    - NLTK for part of speech tagging (uses UPENN treebank)
"""
#%%    
## Starting with textstat  
textstat_results = pd.DataFrame(columns = ['flesch_ease', 'flesch_grade','gfog',
           'auto_readability','cl_index','lw_formula','dcr_score', 
#           'text_standard', 
           'syll_count', 'lex_count'])

for i in clean_data["text"]: #textstat needs a string
    results = textstat_stats(str(i))
    textstat_results = textstat_results.append(results, ignore_index=True) #so that index is continuous

# Resetting indices here may be unneccesary
textstat_results = textstat_results.reset_index(drop=True)

combined_data = pd.concat([clean_data, textstat_results], axis = 1)

## Moving on to NLTK part-of-speech tagging
combined_data_wordtokens = []
for document in combined_data["text"]:
    tokens = nltk.word_tokenize(str(document))
    combined_data_wordtokens.append(tokens)

combined_data_wordpos = []
for document in combined_data_wordtokens:
    pos = nltk.pos_tag(document) #default is Penn Treebank tagset
    combined_data_wordpos.append(pos)
    
#pos_keys = ['CC', 'CD','DT','EX','FW','IN', 'JJ','JJR','JJS','LS','MD','NN','NNS',
#            'NNP','NNPS','PDT','POS','PRP','PRP$','RB','RBR' ,'RBS','RP', 'SYM','TO',
#            'UH','VB', 'VBD', 'VBG','VBN','VBP','VBZ','WDT','WP','WP$','WRB']
from collections import Counter

pos_counts = []

for document in combined_data_wordpos:
    doc_length = len(document)
    mini_dict = Counter([pos for word,pos in document])
    scaled_dict = {k: v / doc_length for k, v in mini_dict.items()}
#    for pos in pos_keys:
#        if pos not in mini_dict:
#            mini_dict[pos] = 0
    pos_counts.append(scaled_dict)

pos_df = pd.DataFrame(pos_counts)
pos_df = pos_df.fillna(0)

combined_data = pd.concat([combined_data, pos_df], axis = 1)
#%%
"""
Add sentiment intensity
"""
#%% 
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA

sia = SIA()
results = []

for line in combined_data["text"]: # this returns a list of dicts
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
    
combined_data["SIA_neg"] = sia_neg
combined_data["SIA_pos"] = sia_pos
combined_data["SIA_neu"] = sia_neu
combined_data["SIA_com"] = sia_comp

#%%
"""
Now for the custom features. 
"""
#%%   

swear_jar = []
what_curses = []
for row in combined_data["text"]:
    num_curses, which_curses = swear_counter(row)    
    swear_jar.append(num_curses)
    what_curses.append(which_curses)
combined_data["num_curses"] = swear_jar
combined_data["which_curses"] = what_curses
combined_data["num_curses"].value_counts()


emoji_counts = []
for row in combined_data["text"]:
    emoji_num = len(emoji_counter(str(row)))
    emoji_counts.append(emoji_num)

combined_data["num_emoji"] = emoji_counts
combined_data["num_emoji"].value_counts()


internet_yelling = []
for row in combined_data["text"]:
    screams = scream_counter(str(row))
    internet_yelling.append(screams)

combined_data["yell_count"] = internet_yelling
combined_data["yell_count"].value_counts()

punct_columns = ['#', '$', "''", '(', ')', ',', '.', ':','``']
combined_data['total_punctuation'] = combined_data.loc[:, punct_columns].sum(axis=1) 

parentheses = ['(',')']
combined_data['parentheses'] = combined_data.loc[:, parentheses].sum(axis=1) 

quotes = ["''",'``', "'"]
combined_data['quotes'] = combined_data.loc[:, quotes].sum(axis=1) 

num_abbreviations = []
for row in combined_data["text"]:
    num_abbrevs = contraction_counter(str(row))
    num_abbreviations.append(num_abbrevs)

combined_data["contractions"] = num_abbreviations

#%%
"""
Visualize features to determine which to include.

Every feature has a bunch of outliers, so let's try masking them
with NaN and then remaking the plots. Don't mask features like yelling and emoji.

That generally looks better,
although the punctuation POS are still maybe unhelpful.
"""
#%% 
features_to_mask = ['flesch_ease', 'flesch_grade', 'gfog',
       'auto_readability', 'cl_index', 'lw_formula', 'dcr_score', 'syll_count',
       'lex_count', '#', '$', "''", '(', ')', ',', '.', ':', 'CC', 'CD', 'DT',
       'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNP', 'NNPS',
       'NNS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM',
       'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$',
       'WRB', '``', 'SIA_neg', 'SIA_pos', 'SIA_neu', 'SIA_com', 'total_punctuation', 
       'parentheses', "contractions", "quotes"]

unmasked_features = ['num_emoji', 'yell_count','num_curses']

# mask the data to hide outliers
masked_data = combined_data.loc[:,features_to_mask].values
mask = np.abs((masked_data - masked_data.mean(0)) / masked_data.std(0)) > 2
masked_data = pd.DataFrame(np.where(mask, np.nan, masked_data), 
                           combined_data.loc[:,features_to_mask].index, 
                           combined_data.loc[:,features_to_mask].columns)

# add category back
masked_data['source'] = combined_data['source']

# add back the unmasked features
combined_data2 = pd.concat([combined_data.iloc[:,0], #the original text
                            combined_data.loc[:,unmasked_features],
                            masked_data], # last column will be source
                            axis = 1)

#combined_data2.loc[combined_data2["num_curses"] >2, ["which_curses"]]

#feature_descriptives = combined_data2.iloc[:,:-1].describe(include='all') #cripes we have terrible variation when you
# look at the whole data set.


# make a ton of barplots to see what the features look like.
#for i in combined_data2.columns[1:-1]:
#    sns.barplot(x = combined_data2["source"], y = combined_data2[i])
#    plt.show()

# make a list of the features with weird distributions or no variance:
features_to_cut = ['lex_count', '#', 'LS', 'SYM','WP$', '``', "''",'(', ')',"quotes"]

combined_data3 = combined_data2.drop(features_to_cut, axis = 1)



#%%
"""
These features have odd distributions and probably need re-engineering:
    - Num_emoji: why so many in reports?
    - Yell_count: why so many in AAM?
    - Num_curses: again, why so many in AAM?
    - flesch_ease: what's a negative reading ease?
    - ": let's combine all the quotation mark features, ''``'
    - (,): these should be combined
    
"""
#%% 




#%%
"""
I tested that the model still performs the same as during prototyping, so now
we'll save the combined_data dataframe via pickle.
"""
#%% 
import xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

# split data into X and y
X = combined_data3.drop(["text", "source"], axis = 1)
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

# fit model on training data
model = XGBClassifier()
model.fit(X_train, y_train)

# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0)) # drops to 78.16% with masked data

fig, ax = plt.subplots(figsize=(4, 10))
xgboost.plot_importance(model, ax=ax)

#xgboost.plot_importance(model)
plt.show()
#%%
"""
Confusion matrix
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
Sanity check: test specific items to see why they were labeled
"""
#%%
#import lime
#import lime.lime_tabular
#
## create lambda function to return probability of the target variable given a set of features
#predict_fn_xgb = lambda x: model.predict_proba(x).astype(float)
##create list of feature names to be used later
#feature_names = combined_data.columns[3:].tolist()
##create LIME explainer
#explainer = lime.lime_tabular.LimeTabularExplainer(X_train, 
#                                                   feature_names = feature_names, 
#                                                   class_names = ['1', '2','3','4','5'],
#                                                   kernel_width = 3)
#
#lime_labled_tuples = list(zip(y_test, predictions))
#lime_labeled_df = pd.DataFrame(lime_labled_tuples, columns = ["true_score", "predicted_score"])
#
#observation_to_check = 392 #used Variable explorer to figure this out
#exp = explainer.explain_instance(X_test[observation_to_check], 
#                                 predict_fn_xgb, 
#                                 num_features = len(feature_names))
#
#import pickle
#filename = '../pickled_LIME_5_as_5.sav'
#pickle.dump(exp, open(filename, 'wb'))
#%%
"""
Further sanity check: using Tree SHAP instead of LIME
"""
#%%
import shap

X_train_labeled = pd.DataFrame(X_train, columns = X.columns)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train_labeled)
shap.summary_plot(shap_values, X_train_labeled)

#shap.summary_plot(shap_values[0], X_train_labeled)
#shap.force_plot(explainer.expected_value[0], shap_values[0], X_train_labeled.iloc[0,:])

#%%
"""
I tested that the model still performs the same as during prototyping, so now
we'll save the combined_data dataframe via pickle. 
"""
#%% 
import pickle
filename = '../pickled_corpus_w_features.sav'
pickle.dump(combined_data, open(filename, 'wb'))

#%%