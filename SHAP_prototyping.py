#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 20:31:27 2019

@author: amandakomuro
"""
#%%
import pandas as pd
import numpy as np
import csv
import pickle
from collections import Counter
import nltk
import textstat as ts
import emoji
import regex
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA


model = pickle.load(open("./insight2019/flask_app/my_flask/model/finalized_XGBoost_model.sav", 'rb'))
scaler = pickle.load(open("./insight2019/flask_app/my_flask/model/finalized_XGBoost_scaler.sav", 'rb'))
final_column_order = pickle.load(open("./insight2019/flask_app/my_flask/model/finalized_column_order.sav", 'rb'))

source_averages = pickle.load(open("./insight2019/flask_app/my_flask/model/finalized_source_averages.sav", 'rb'))

#%%

def textstat_stats(text):
    doc_length = len(text.split()) 
    flesch_ease = ts.flesch_reading_ease(text) #Flesch Reading Ease Score
    flesch_grade = ts.flesch_kincaid_grade(text) #Flesch-Kincaid Grade Level
    gfog = ts.gunning_fog(text) # FOG index, also indicates grade level
    auto_readability = ts.automated_readability_index(text) #approximates the grade level needed to comprehend the text.
    cl_index = ts.coleman_liau_index(text) #grade level of the text using the Coleman-Liau Formula.
    lw_formula = ts.linsear_write_formula(text) #grade level using the Linsear Write Formula.
    dcr_score = ts.dale_chall_readability_score(text) #uses a lookup table of the most commonly used 3000 English words
    syll_count = ts.syllable_count(text, lang='en_US')
    syll_count_scaled = syll_count / doc_length
    lex_count = ts.lexicon_count(text, removepunct=True)
    lex_count_scaled = lex_count / doc_length
    idx = ['flesch_ease', 'flesch_grade','gfog',
           'auto_readability','cl_index','lw_formula',
           'dcr_score', 
           'syll_count', 'lex_count']
    return pd.Series([flesch_ease, flesch_grade, gfog, 
                      auto_readability, cl_index, lw_formula, 
                      dcr_score, 
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



def process_user_text(user_text, goal_category):
    #put user input text string into a DataFrame
    clean_data = pd.DataFrame(user_text, columns = ["text"]) 
    clean_data["source"] = goal_category
    clean_data["subreddit"] = "placeholder"

    ## Starting with textstat  
    textstat_results = pd.DataFrame(columns = ['flesch_ease', 'flesch_grade','gfog',
               'auto_readability','cl_index','lw_formula','dcr_score', 
               'syll_count', 'lex_count'])

    for i in clean_data["text"]: 
        results = textstat_stats(str(i)) #textstat needs a string
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
        
    pos_keys = ['#', '$', '“', '(', ')', ',', '.', ':', 'CC', 'CD', 'DT', 'EX', 
                'FW', 'IN', 'JJ', 'JJR', 'JJS','LS', 'MD', 'NN', 'NNP', 'NNPS', 
                'NNS', 'PDT', 'POS', 'PRP', 'PRP$','RB', 'RBR', 'RBS', 'RP', 
                'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN','VBP', 'VBZ', 
                'WDT', 'WP', 'WP$', 'WRB', '”']

    pos_counts = []

    for document in combined_data_wordpos:
        doc_length = len(document)
        mini_dict = Counter([pos for word,pos in document])
        for pos in pos_keys:
            if pos not in mini_dict:
                mini_dict[pos] = 0
        scaled_dict = {k: v / doc_length for k, v in mini_dict.items()}
        pos_counts.append(scaled_dict)

    pos_df = pd.DataFrame(pos_counts)
    pos_df = pos_df.fillna(0)

    combined_data = pd.concat([combined_data, pos_df], axis = 1)

    ## Add sentiment intensity
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


    ## Now for the custom features
    Google_Curses = pd.read_csv("./insight2019/flask_app/my_flask/model/RobertJGabriel_Google_swear_words.txt", header = None)
    bad_words = Google_Curses[0].tolist()

    any_bad = []
    for row in combined_data["text"]:
        if any(str(word) in str(row) for word in bad_words):
            any_bad.append(1)
        else: any_bad.append(0)

    combined_data["Google_curses"] = any_bad
    combined_data["Google_curses"].value_counts()

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

    return combined_data


def get_professionalism_score(user_df):
    combined_data = user_df

    # split data into X and y
    userX = combined_data.iloc[:,3:] #columns 0-2 are the text and categories
    userY = combined_data['source'] # "source" is the column of numeric sources

    # scale the data using scaler trained on original corpus
    userX = scaler.transform(userX) 

    # make predictions for test data
    y_probs = model.predict_proba(userX)
    
    top_two_classes = y_probs[0].argsort()[-2:][::-1]
    first_class_prob = y_probs[0][top_two_classes[0]]
    first_class_prob  = round(first_class_prob * 100)
    second_class_prob = y_probs[0][top_two_classes[1]]
    second_class_prob  = round(second_class_prob * 100)

    #attach column names to scaled features for user
    userX_df = pd.DataFrame(userX)
    userX_df.columns = final_column_order[3:]
    userX_df = userX_df.round(decimals = 2)
    userX_df = userX_df.reset_index(drop=True)

    # class_diff = first_class_prob - second_class_prob 

    # if class_diff > 0.2: # if the choice was clear, go with highest prob class
    #     user_prof_score = top_two_classes[0]
    # else:  # if choice was close, go with a value in between
    #     user_prof_score = min(top_two_classes[0], top_two_classes[1]) + 0.5
    
    # return user_prof_score
    return top_two_classes, first_class_prob, second_class_prob, userX_df, y_probs
#    return top_two_classes, first_class_prob, second_class_prob
#    return y_probs

#%%
user_text = "Understanding why a model makes a certain prediction can be as crucial as the prediction’s accuracy in many applications. However, the highest accuracy for large modern datasets is often achieved by complex models that even experts struggle to interpret, such as ensemble or deep learning models, creating a tension between accuracy and interpretability. In response, various methods have recently been proposed to help users interpret the predictions of complex models, but it is often unclear how these methods are related and when one method is preferable over another. To address this problem, we present a unified framework for interpreting predictions, SHAP (SHapleyAdditive exPlanations). SHAP assigns each feature an importance value for a particular prediction. Its novel components include: (1)the identification of a new class of additive feature importance measures, and (2)theoretical results showing there is a unique solution in this class with a set of desirable properties. "

# read data (should be a string) and make it a list of length 1.
# user_text1 = [i for i in user_text]
user_text1 = [user_text]

goal_category = "Extremely Casual"

user_df = process_user_text(user_text1, goal_category) #DataFrame with 1 row

# run the model function
top_two_classes, first_class_prob, second_class_prob, userX, y_probs = get_professionalism_score(user_df)
#top_two_classes, first_class_prob, second_class_prob = get_professionalism_score(user_df)

#y_probs = get_professionalism_score(user_df)

#%%
"""
Plot prototyping below
"""
#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def make_classification_plot(y_probs, top_class_numeric): 
    y_probs2 = [i for i in y_probs[0]]
    df = pd.DataFrame(y_probs2, columns = ["probs"])
    df["source"] = source_to_id
    df["source2"] = pd.Categorical(df["source"])
    my_range=range(1,len(df.index)+1)
 
    # Create a color if the source is the highest probability class
    my_color=np.where(df ['source2']==id_to_source[top_class_numeric], 'orange', 'skyblue')
    my_size=np.where(df ['source2']==id_to_source[top_class_numeric], 70, 30)
     
    # The vertival plot is made using the hline function
    plt.figure(figsize=[6,2])
    plt.tight_layout()
    
    plt.hlines(y=my_range, xmin=0, xmax=df['probs']*100, color=my_color, alpha=0.4)
    plt.scatter(df['probs']*100, my_range, color=my_color, s=my_size, alpha=1)
     
    # Add title and exis names
    plt.yticks(my_range, df['source'])
    plt.title("Percent match to each category", loc='left')
    plt.xlabel(None)
    plt.ylabel(None)
 
    plt.savefig('./insight2019/flask_app/my_flask/model/test_plot.png', bbox_inches='tight')
    
    

id_to_source = {0: 'Extremely Casual', 1: 'Company IM', 2: 'Workplace Casual', 3: 'Reports', 4: 'Dissertations'}
source_to_id = {'Extremely Casual': 0 ,'Company IM': 1, 'Workplace Casual': 2, 'Reports': 3, 'Dissertations': 4}


make_classification_plot(y_probs, top_two_classes[0])
#
#y_probs2 = [i for i in y_probs[0]]
#df = pd.DataFrame(y_probs2, columns = ["probs"])
#df["source"] = source_to_id
#df["source2"] = pd.Categorical(df["source"])
#
#
## libraries
#import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
# 
## Create a dataframe 
#my_range=range(1,len(df.index)+1)
# 
## Create a color if the source is the highest probability class
#my_color=np.where(df ['source2']==id_to_source[top_two_classes[0]], 'orange', 'skyblue')
#my_size=np.where(df ['source2']==id_to_source[top_two_classes[0]], 70, 30)
# 
## The vertival plot is made using the hline function
#plt.hlines(y=my_range, xmin=0, xmax=df['probs']*100, color=my_color, alpha=0.4)
#plt.scatter(df['probs']*100, my_range, color=my_color, s=my_size, alpha=1)
# 
## Add title and exis names
#plt.yticks(my_range, df['source'])
#plt.title("Percent match to each category", loc='left')
#plt.xlabel(None)
#plt.ylabel(None)



#%%
"""
SHAP issues
"""
#%%
import shap

def get_SHAP_results(user_df, user_category, goal_category):
    
    with open("./insight2019/flask_app/my_flask/model/column_keys.txt") as fin:
         rows = ( line.strip().split('\t') for line in fin )
         column_dict = { row[0]:row[1:] for row in rows }
    
    
    #starting with output from process_user_text()
    X_train_labeled = pd.DataFrame(user_df, columns = user_df.columns[3:])
    #set up the SHAP explainer for the XGBoost model and calc SHAP scores
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train_labeled) # list of lists, outer lists
    # are the categories, inner list is feature contributions to that category
    
    # re-arrange SHAP values so they're easy to give back to user
    source0_shap=pd.DataFrame(data=shap_values[0], columns=user_df.columns[3:], index=[0])
    source1_shap=pd.DataFrame(data=shap_values[1], columns=user_df.columns[3:], index=[1])
    source2_shap=pd.DataFrame(data=shap_values[2], columns=user_df.columns[3:], index=[2])
    source3_shap=pd.DataFrame(data=shap_values[3], columns=user_df.columns[3:], index=[3])
    source4_shap=pd.DataFrame(data=shap_values[4], columns=user_df.columns[3:], index=[4])
    
    user_reasoning=pd.concat([source0_shap,
                              source1_shap,
                              source2_shap,
                              source3_shap,
                              source4_shap])
    user_reasoning = user_reasoning.transpose()
    
    # make sure to translate the written sources into numeric
    goal_numeric = source_to_id[goal_category]
    
    #Why did user text get it's particular score?
    user_category_reasoning = user_reasoning.sort_values(by=[user_category], ascending=False)
    user_category_reasoning_labels = user_category_reasoning.index.values[0:3]
    user_category_reasoning_labels2 = [column_dict[feature] for feature in user_category_reasoning_labels] 
    user_category_reasoning_labels2 = [item for sublist in user_category_reasoning_labels2 for item in sublist]
    user_category_reasoning_avgs = source_averages.loc[user_category, user_category_reasoning_labels]
    
    # What features diminished their odds of being labeled as their goal?
    user_goal_improvement = user_reasoning.sort_values(by=[goal_numeric], ascending=True)
    user_goal_improvement_labels = user_goal_improvement.index.values[0:3]
    user_goal_improvement_labels2 = [column_dict[feature] for feature in user_goal_improvement_labels] 
    user_goal_improvement_labels2 = [item for sublist in user_goal_improvement_labels2 for item in sublist]
    user_goal_improvement_avgs = source_averages.loc[goal_numeric, user_goal_improvement_labels]


    # What features improved their odds of being labeled as their goal?
    user_goal_encouragement = user_reasoning.sort_values(by=[goal_numeric], ascending=False)
    user_goal_encouragement_labels = user_goal_encouragement.index.values[0:3]
    user_goal_encouragement_labels2 = [column_dict[feature] for feature in user_goal_encouragement_labels]
    user_goal_encouragement_labels2 = [item for sublist in user_goal_encouragement_labels2 for item in sublist]
    user_goal_encouragement_avgs = source_averages.loc[goal_numeric, user_goal_encouragement_labels]

    
    return user_category_reasoning_labels, user_category_reasoning_labels2, user_category_reasoning_avgs, user_goal_improvement_labels, user_goal_improvement_labels2, user_goal_improvement_avgs, user_goal_encouragement_labels, user_goal_encouragement_labels2, user_goal_encouragement_avgs


#%%
"""
Making lollipop
"""
#%%


category_reasoning, category_reasoning2, category_avgs, improvement_feedback, improvement_feedback2, improvement_avgs, encouragement_feedback, encouragement_feedback2, encouragement_avgs = get_SHAP_results(user_df, top_two_classes[0], goal_category)

user_cat_val = userX.loc[0, category_reasoning]
imp_cat_val = userX.loc[0, improvement_feedback]
enc_cat_val = userX.loc[0, encouragement_feedback]


def make_feedback_lollipop(improvement_feedback2, imp_cat_val, improvement_avgs, encouragement_feedback2, enc_cat_val, encouragement_avgs):
    lollipop_red = pd.DataFrame([improvement_feedback2, imp_cat_val, improvement_avgs])
    lollipop_red = lollipop_red.transpose()
    lollipop_red.columns = ["feature_name", "user_score", "category_average"]
    lollipop_red["purpose"] = "critique"
    
    lollipop_green  = pd.DataFrame([encouragement_feedback2, enc_cat_val, encouragement_avgs])
    lollipop_green = lollipop_green.transpose()
    lollipop_green.columns = ["feature_name", "user_score", "category_average"]
    lollipop_green["purpose"] = "encouragement"
    
    lollipops = pd.concat([lollipop_green, lollipop_red], axis = 0)
    lollipops["user_diff"] = lollipops["user_score"] - lollipops["category_average"]
    
    # add a threshold so that plots stay interpretable
    lollipops["user_diff"] = lollipops["user_diff"].clip(-2,2)
    
    # Create a color if the source is the highest probability class
    my_range=range(1,len(lollipops.index)+1)
    my_color=np.where(lollipops ['purpose']=="critique", 'purple', 'green')
    #my_size=np.where(lollipops ['source2']==id_to_source[top_class_numeric], 70, 30)
     
    # The vertival plot is made using the hline function
    fig, ax = plt.subplots(figsize=(6, 2.4))
    #plt.figure(figsize=[3,6])# longer than wide
    plt.tight_layout()
    
    plt.hlines(y=my_range, xmin=0, xmax=lollipops['user_diff'], color=my_color, alpha=0.4)
    plt.vlines(x=0, ymin = min(my_range), ymax=max(my_range))
    plt.scatter(lollipops['user_diff'], 
                my_range, 
                color=my_color, 
                alpha=1)
     
    # Add title and exis names
    plt.yticks(my_range, lollipops['feature_name'])
    plt.title("Difference from goal category average", loc='left')
    plt.xlabel(None)
    plt.ylabel(None)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.get_xaxis().set_visible(False)

 
