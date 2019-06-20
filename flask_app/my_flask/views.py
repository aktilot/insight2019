#!/usr/bin/env python
from flask import Flask, render_template, request
import requests
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


#Initialize app
app = Flask(__name__, static_url_path='/static')


### Load the pickled objects for my model and other needed parts.
model = pickle.load(open("./model/finalized_XGBoost_model.sav", 'rb'))
scaler = pickle.load(open("./model/finalized_XGBoost_scaler.sav", 'rb'))

id_to_source = {0: 'Extremely Casual',1:'Company IM', 2:'Workplace Casual', 3:'Reports', 4:'Dissertations'}


"""
Custom functions for feature engineering
"""

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
    Google_Curses = pd.read_csv("./model/RobertJGabriel_Google_swear_words.txt", header = None)
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
    second_class_prob = y_probs[0][top_two_classes[1]]
    class_diff = first_class_prob - second_class_prob 

    if class_diff > 0.2: # if the choice was clear, go with highest prob class
        user_prof_score = top_two_classes[0]
    else:  # if choice was close, go with a value in between
        user_prof_score = min(top_two_classes[0], top_two_classes[1]) + 0.5
    
    return user_prof_score



#Standard home page. 'index.html' is the file in your templates that has the CSS and HTML for your app
# @app.route('/', methods=['GET', 'POST'])
# def index():
#     return render_template('index.html')

# @app.route('/input', methods=['GET', 'POST'])
 
@app.route('/', methods=['GET', 'POST'])
def get_inputs():
  sources = list(id_to_source.values())
  return render_template("inputs.html", goal_category = sources)


@app.route('/output', methods=['GET', 'POST'])
def get_outputs():

    goal_category = str(request.form['goal_category'])
    user_text = str(request.form['user_text'])
    # user_text = "This is a placeholder sentence."

    # read data (should be a string) and make it a list of length 1.
    # user_text1 = [i for i in user_text]
    user_text1 = [user_text]

    user_df = process_user_text(user_text1, goal_category) # Returns a DataFrame with 1 row

    # run the model function
    # prof_score = "placeholder"
    prof_score = get_professionalism_score(user_df)

    return render_template(
      "outputs.html", 
      values = { 
          'user_text': user_text if user_text else "No input.",
          'goal_category': goal_category if goal_category else "No input.",
          'prof_score': prof_score
      })


# 'prof_score': prof_score if prof_score else "No result."



# @app.route('/input')
# def get_inputs():
#     print(request.args)

#     user_text = request.args.get('user_text')
#     # user_text = "This is a placeholder sentence."
#     prof_score = "placeholder"

#     # read data (should be a string) and make it a list.
#     user_text1 = [i for i in user_text]

#     # run the model function
#     prof_score = get_professionalism_score(user_text1)

#     return render_template(
#       "inputs.html",
#        title = 'Home', 
#        values = { 
#           'user_text': user_text if user_text else "No input.",
#           'prof_score': prof_score if prof_score else "No result."
#       },
#     )


if __name__ == '__main__':
    #this runs your app locally
    app.run(host='0.0.0.0', port=8080, debug=True)