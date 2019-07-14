#!/usr/bin/env python
from flask import Flask

from flask import render_template, request
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
import shap
from io import BytesIO
import base64
import urllib.parse
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost

#Initialize app
# application = Flask(__name__, static_url_path='/static')
app = Flask(__name__, static_url_path='/static')

### Load the pickled objects for my model and other needed parts.
model = pickle.load(open("./model/finalized_XGBoost_model2.sav", 'rb'))
scaler = pickle.load(open("./model/finalized_XGBoost_scaler2.sav", 'rb'))
final_column_order = pickle.load(open("./model/finalized_column_order2.sav", 'rb'))
source_averages = pickle.load(open("./model/finalized_source_averages2.sav", 'rb'))
source_averages = source_averages.round(decimals = 2)
source_averages = source_averages.reset_index(drop=True)
source_averages.columns = final_column_order[1:55]

id_to_source = {0: 'Extremely Casual', 1: 'Company IM', 2: 'Workplace Casual', 3: 'Reports', 4: 'Dissertations'}
source_to_id = {'Extremely Casual': 0 ,'Company IM': 1, 'Workplace Casual': 2, 'Reports': 3, 'Dissertations': 4}


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


Google_Curses = pd.read_csv("./model/custom_curse_words.txt", header = None)
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

"""
Process the user text, to generate features used in model
"""
def process_user_text(user_text, goal_category):

    ## put user input text string into a DataFrame
    clean_data = pd.DataFrame(user_text, columns = ["text"]) 
    clean_data["source"] = goal_category
    # clean_data["subreddit"] = "placeholder"

    # remove hyperlinks & bullets (so they don't get counted as emoji)
    clean_data["text"] = clean_data["text"].str.replace(r'http\S*\s', ' ')
    clean_data["text"] = clean_data["text"].str.replace(r'http\S*(\n|\)|$)', ' ')
    clean_data["text"] = clean_data["text"].str.replace(r'', ' ')

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

    pos_counts = []
    
    pos_keys = ['CC', 'CD','DT','EX','FW','IN', 'JJ','JJR','JJS','LS','MD','NN','NNS',
               'NNP','NNPS','PDT','POS','PRP','PRP$','RB','RBR' ,'RBS','RP', 'SYM','TO',
               'UH','VB', 'VBD', 'VBG','VBN','VBP','VBZ','WDT','WP','WP$','WRB',
               '#', '$', "''", '(', ')', ',', '.', ':','``']


    for document in combined_data_wordpos:
        doc_length = len(document)
        mini_dict = Counter([pos for word,pos in document])
        for pos in pos_keys:
           if pos not in mini_dict:
               mini_dict[pos] = 0
        scaled_dict = {k: v / doc_length for k, v in mini_dict.items()}
        pos_counts.append(scaled_dict)

    # for document in combined_data_wordpos:
    #     doc_length = len(document)
    #     mini_dict = Counter([pos for word,pos in document])
    #     scaled_dict = {k: v / doc_length for k, v in mini_dict.items()}
    #     pos_counts.append(scaled_dict)

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
    #  cursing
    swear_jar = []
    what_curses = []
    for row in combined_data["text"]:
        num_curses, which_curses = swear_counter(row)    
        swear_jar.append(num_curses)
        what_curses.append(which_curses)
    combined_data["num_curses"] = swear_jar

    # emoji use
    emoji_counts = []
    for row in combined_data["text"]:
        emoji_num = len(emoji_counter(str(row)))
        emoji_counts.append(emoji_num)
    combined_data["num_emoji"] = emoji_counts

    # !?,  ?!, ??, and !!
    internet_yelling = []
    for row in combined_data["text"]:
        screams = scream_counter(str(row))
        internet_yelling.append(screams)
    combined_data["yell_count"] = internet_yelling

    # Combine the punctuation columns to get a total measure
    punct_columns = ['#', '$', "''", '(', ')', ',', '.', ':','``']
    combined_data['total_punctuation'] = combined_data.loc[:, punct_columns].sum(axis=1) 

    # Combine parentheses
    parentheses = ['(',')']
    combined_data['parentheses'] = combined_data.loc[:, parentheses].sum(axis=1) 

    # Encoding of quotation marks varies by source, need to combine them.
    quotes = ["''",'``', "'"]
    combined_data['quotes'] = combined_data.loc[:, quotes].sum(axis=1) 

    # Count up contractions, scale by doc length
    num_abbreviations = []
    for row in combined_data["text"]:
        num_abbrevs = contraction_counter(str(row))
        num_abbreviations.append(num_abbrevs)
    combined_data["contractions"] = num_abbreviations


    # Google_Curses = pd.read_csv("./model/custom_curse_words.txt", header = None)
    # bad_words = Google_Curses[0].tolist()

    # any_bad = []
    # for row in combined_data["text"]:
    #     if any(str(word) in str(row) for word in bad_words):
    #         any_bad.append(1)
    #     else: any_bad.append(0)

    # combined_data["Google_curses"] = any_bad
    # combined_data["Google_curses"].value_counts()

    # emoji_counts = []
    # for row in combined_data["text"]:
    #     emoji_num = len(emoji_counter(str(row)))
    #     emoji_counts.append(emoji_num)

    # combined_data["Num_emoji"] = emoji_counts
    # combined_data["Num_emoji"].value_counts()

    # internet_yelling = []
    # for row in combined_data["text"]:
    #     screams = scream_counter(str(row))
    #     internet_yelling.append(screams)

    # combined_data["Yell_count"] = internet_yelling

    ## Make sure user data is in the same order as the model.
    features_to_drop = ['gfog','``','auto_readability','lex_count', '#', 
                        'LS', 'SYM','WP$', '``', "''",'(', ')']

    combined_data = combined_data.drop(features_to_drop, axis = 1)

    # this may be extremely important. starts with text, ends with source
    combined_data = combined_data[final_column_order] 

    return combined_data


def get_professionalism_score(user_df):
    combined_data = user_df

    # split data into X and y
    userX = combined_data.drop(["text", "source"], axis = 1)
    userY = combined_data['source'] # "source" is the goal category

    col_names = userX.columns # will user later to reattach names

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
    userX_df.columns = col_names
    userX_df = userX_df.round(decimals = 2)
    userX_df = userX_df.reset_index(drop=True)

    # class_diff = first_class_prob - second_class_prob 

    # if class_diff > 0.2: # if the choice was clear, go with highest prob class
    #     user_prof_score = top_two_classes[0]
    # else:  # if choice was close, go with a value in between
    #     user_prof_score = min(top_two_classes[0], top_two_classes[1]) + 0.5
    
    # return user_prof_score
    return top_two_classes, first_class_prob, second_class_prob, userX_df, y_probs


def make_classification_plot(y_probs, top_class_numeric): 
    y_probs2 = [i for i in y_probs[0]]
    df = pd.DataFrame(y_probs2, columns = ["probs"])
    df["source"] = source_to_id
    df["source2"] = pd.Categorical(df["source"])
    my_range=range(1,len(df.index)+1)
 
    # Create a color if the source is the highest probability class
    my_color=np.where(df ['source2']==id_to_source[top_class_numeric], 'orange', 'skyblue')
    my_size=np.where(df ['source2']==id_to_source[top_class_numeric], 70, 30)
    
    # set up the thing that will hold my figure 
    img = BytesIO()

    # image size 
    fig, ax = plt.subplots(figsize=[6,2], dpi = 300)
    plt.tight_layout()

    # The vertival plot is made using the hline function
    plt.hlines(y=my_range, xmin=0, xmax=df['probs']*100, color=my_color, alpha=0.4)
    plt.scatter(df['probs']*100, my_range, color=my_color, s=my_size, alpha=1)
     
    # Add title and exis names
    plt.yticks(my_range, df['source'])
    # plt.title(, loc='left')
    plt.xlabel("Percent match to each category")
    plt.ylabel(None)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # add the plot data to the img thing
    plt.savefig(img, format='png', bbox_inches = 'tight')
    plt.close()
    img.seek(0)
    img_png_target = img.getvalue()
    plot_url = base64.b64encode(img_png_target)

    return plot_url
    # plt.savefig('./insight2019/flask_app/my_flask/model/test_plot.png', bbox_inches='tight')

def get_SHAP_results(user_df, user_category, goal_category):
    
    #set up dictionary of feature explanations.
    with open("./model/column_keys.txt") as fin:
         rows = ( line.strip().split('\t') for line in fin )
         column_dict = { row[0]:row[1:] for row in rows }
    
    
    #starting with output from process_user_text()
    # X_train_labeled = pd.DataFrame(user_df, columns = user_df.columns[3:])
    user_df = user_df.drop(["text", "source"], axis = 1) # now we have only feature columns
    X_train_labeled = user_df

    #set up the SHAP explainer for the XGBoost model and calc SHAP scores
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train_labeled) # list of lists, outer lists
    # are the categories, inner list is feature contributions to that category
    
    # re-arrange SHAP values so they're easy to give back to user
    # source0_shap=pd.DataFrame(data=shap_values[0], columns=user_df.columns[3:], index=[0])
    # source1_shap=pd.DataFrame(data=shap_values[1], columns=user_df.columns[3:], index=[1])
    # source2_shap=pd.DataFrame(data=shap_values[2], columns=user_df.columns[3:], index=[2])
    # source3_shap=pd.DataFrame(data=shap_values[3], columns=user_df.columns[3:], index=[3])
    # source4_shap=pd.DataFrame(data=shap_values[4], columns=user_df.columns[3:], index=[4])
    source0_shap=pd.DataFrame(data=shap_values[0], columns=user_df.columns, index=[0])
    source1_shap=pd.DataFrame(data=shap_values[1], columns=user_df.columns, index=[1])
    source2_shap=pd.DataFrame(data=shap_values[2], columns=user_df.columns, index=[2])
    source3_shap=pd.DataFrame(data=shap_values[3], columns=user_df.columns, index=[3])
    source4_shap=pd.DataFrame(data=shap_values[4], columns=user_df.columns, index=[4])
    
    user_reasoning=pd.concat([source0_shap,
                              source1_shap,
                              source2_shap,
                              source3_shap,
                              source4_shap])
    user_reasoning = user_reasoning.transpose()
    
    # make sure to translate the written sources into numeric
    goal_numeric = source_to_id[goal_category]
#    user_cat_string= id_to_source[user_category] #testing
    
    #Why did user text get it's particular score? (top 3 features)
#    user_category_reasoning = user_reasoning.sort_values(by=[user_cat_string], ascending=False)
    user_category_reasoning = user_reasoning.sort_values(by=user_category, ascending=False)
    user_category_reasoning_labels = user_category_reasoning.index.values[0:3]
    user_category_reasoning_labels2 = [column_dict[feature] for feature in user_category_reasoning_labels] 
    user_category_reasoning_labels2 = [item for sublist in user_category_reasoning_labels2 for item in sublist]
    user_category_reasoning_avgs = source_averages.loc[user_category, user_category_reasoning_labels]
    
    # What features diminished their odds of being labeled as their goal?  (top 3 features)
    user_goal_improvement = user_reasoning.sort_values(by=[goal_numeric], ascending=True)
    user_goal_improvement_labels = user_goal_improvement.index.values[0:3]
    user_goal_improvement_labels2 = [column_dict[feature] for feature in user_goal_improvement_labels] 
    user_goal_improvement_labels2 = [item for sublist in user_goal_improvement_labels2 for item in sublist]
    user_goal_improvement_avgs = source_averages.loc[goal_numeric, user_goal_improvement_labels]


    
    # What features improved their odds of being labeled as their goal?  (top 3 features)
    user_goal_encouragement = user_reasoning.sort_values(by=[goal_numeric], ascending=False)
    user_goal_encouragement_labels = user_goal_encouragement.index.values[0:3]
    user_goal_encouragement_labels2 = [column_dict[feature] for feature in user_goal_encouragement_labels]
    user_goal_encouragement_labels2 = [item for sublist in user_goal_encouragement_labels2 for item in sublist]
    user_goal_encouragement_avgs = source_averages.loc[goal_numeric, user_goal_encouragement_labels]

    
    return user_category_reasoning_labels, user_category_reasoning_labels2, user_category_reasoning_avgs, user_goal_improvement_labels, user_goal_improvement_labels2, user_goal_improvement_avgs, user_goal_encouragement_labels, user_goal_encouragement_labels2, user_goal_encouragement_avgs



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
    
    # set up the thing that will hold my figure 
    img = BytesIO()

    # The vertival plot is made using the hline function
    fig, ax = plt.subplots(figsize=(6, 2.4), dpi = 300)
    plt.tight_layout()
    
    plt.vlines(x=0, ymin = min(my_range), ymax=max(my_range), linestyles="dashed", color = "silver", zorder=0)
    plt.hlines(y=my_range, xmin=0, xmax=lollipops['user_diff'], color=my_color, alpha=0.4)
    plt.scatter(lollipops['user_diff'], 
                my_range, 
                color=my_color, 
                alpha=1)
     
    # Add title and exis names
    plt.yticks(my_range, lollipops['feature_name'])
    # plt.title(, loc='left')
    plt.xlabel("Difference from goal category average")
    plt.ylabel(None)
    plt.xlim(-2,2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.set_xticks([-2,0,2])
    plt.xticks([-2,0,2], ('Too low', 'Just right', 'Too high'))
    # ax.get_xaxis().set_visible(False)

    # add the plot data to the img thing
    plt.savefig(img, format='png', bbox_inches = 'tight')
    plt.close()
    img.seek(0)
    img_png_target = img.getvalue()
    plot_url = base64.b64encode(img_png_target)

    return plot_url



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

    # process the text, get back a dataframe
    user_df = process_user_text(user_text1, goal_category) # Returns a DataFrame with 1 row

    # run the model function
    top_two_classes, first_class_prob, second_class_prob, userX, y_probs = get_professionalism_score(user_df)
    top_class = id_to_source[top_two_classes[0]]
    second_class = id_to_source[top_two_classes[1]]

    # make a plot to show probability of classification in each category
    class_plot = make_classification_plot(y_probs, top_two_classes[0])


    # Run SHAP, get 3 lists of 3 features each
    # For those features, look up their average for either the goal or user category
    category_reasoning, category_reasoning2, category_avgs, improvement_feedback, improvement_feedback2, improvement_avgs, encouragement_feedback, encouragement_feedback2, encouragement_avgs = get_SHAP_results(user_df, top_two_classes[0], goal_category)

    # Get user values for all 9 features I want to report
    user_cat_val = userX.loc[0, category_reasoning]
    imp_cat_val = userX.loc[0, improvement_feedback]
    enc_cat_val = userX.loc[0, encouragement_feedback]

    # make a results plot that summarizes the good and bad
    feedback_plot = make_feedback_lollipop(improvement_feedback2, imp_cat_val, improvement_avgs, encouragement_feedback2, enc_cat_val, encouragement_avgs)


    return render_template(
      "outputs.html", 
      class_plot_url = urllib.parse.quote(class_plot),
      feedback_plot_url = urllib.parse.quote(feedback_plot),
      values = { 
          'user_text': user_text if user_text else "No input.",
          'goal_category': goal_category if goal_category else "No input.",
          # 'class_plot': class_plot,
          'top_class': top_class,
          'second_class': second_class,
          'top_class_prob': first_class_prob,
          'second_class_prob': second_class_prob,
          'category_reasoning': category_reasoning2,
          'user_cat_val': user_cat_val,
          'category_avgs': category_avgs,
          'improvement_feedback': improvement_feedback2,
          'imp_cat_val': imp_cat_val,
          'improvement_avgs': improvement_avgs,
          'encouragement_feedback': encouragement_feedback2,
          'enc_cat_val': enc_cat_val,
          'encouragement_avgs': encouragement_avgs
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
    # this runs your app locally:
    # application.run(host='0.0.0.0', port=8080, debug=True)
    app.run(host='0.0.0.0',debug=True)