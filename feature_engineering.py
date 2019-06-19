#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%
import pandas as pd
import numpy as np
#%%
"""
Custom functions for feature engineering
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
    text_standard = textstat.text_standard(text, float_output=False) # summary of all the grade level functions
    syll_count = ts.syllable_count(text, lang='en_US')
    lex_count = ts.lexicon_count(text, removepunct=True)
    idx = ['flesch_ease', 'flesch_grade','gfog',
           'auto_readability','cl_index','lw_formula',
           'dcr_score', 'text_standard', 'syll_count', 'lex_count']
    return pd.Series([flesch_ease, flesch_grade, gfog, 
                      auto_readability, cl_index, lw_formula, 
                      dcr_score, text_standard, syll_count, lex_count], index = idx)


import emoji
import regex

def emoji_counter(text):
    emoji_list = []
    data = regex.findall(r'\X', text)
    for word in data:
        if any(char in emoji.UNICODE_EMOJI for char in word):
            emoji_list.append(word)
    return emoji_list


import re
def scream_counter(text):
    crazy_confused = len(re.findall(r'!\?|\?!', text))
    ahhhh = len(re.findall(r'!!', text))
    huhhh = len(re.findall(r'\?\?', text))
    screams = crazy_confused + ahhhh + huhhh            
    return screams
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
        
# recode the sources (needed for model?)
sources_dict = {'Extremely Casual': 1, 'Slack-like': 2, 'Workplace_Casual': 3, 
                'Governmental': 4, 'Dissertation': 5}
clean_data.replace(sources_dict, inplace=True)        

# removing rows without a source
clean_data = clean_data.dropna(subset=['source'], axis = 0) #remove 1 weird row
clean_data["source"].value_counts()

# maybe also remove rows that end up with nothing left in the text?
#clean_data = clean_data[clean_data["text"] != ' ']
#clean_data = clean_data[clean_data["text"] != '']
#%%
"""
First create features from NLP packages:
    - textstat for readability measures (they're not all the same)
    - NLTK for part of speech tagging (uses UPENN treebank)
"""
#%%    
## Starting with textstat  
textstat_results = pd.DataFrame(columns = ['flesch_ease', 'flesch_grade','gfog',
           'auto_readability','cl_index','lw_formula','dcr_score', 'text_standard', 'syll_count', 'lex_count'])

for i in clean_data["text"]: #textstat needs a string
    results = textstat_stats(str(i))
    textstat_results = textstat_results.append(results, ignore_index=True) #so that index is continuous

# Resetting indices here may be unneccesary
clean_data = clean_data.reset_index(drop=True)
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

combined_data = pd.concat([combined_data,pos_df], axis = 1)
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
Now for the custom features
"""
#%%   
Google_Curses = pd.read_csv("./data/RobertJGabriel_Google_swear_words.txt", header = None)
bad_words = Google_Curses[0].tolist()

any_bad = []
for row in combined_data["text"]:
    if any(str(word) in str(row) for word in bad_words):
        any_bad.append(1)
    else: any_bad.append(0)

combined_data["Google_curses"] = any_bad
combined_data["Google_curses"].value_counts() #much better, only 5817 with a curse


emoji_counts = []
for row in combined_data["text"]:
    emoji_num = len(emoji_counter(str(row)))
    emoji_counts.append(emoji_num)

combined_data["Num_emoji"] = emoji_counts
combined_data["Num_emoji"].value_counts()


internet_yelling = []
for row in combined_data["text"]:
    screams = scream_counter(str(row))
    internet_yelling.append(screams)

combined_data["Yell_count"] = internet_yelling
combined_data["Yell_count"].value_counts()





