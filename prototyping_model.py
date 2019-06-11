#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
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
enron = pd.read_csv("../data/enron_05_17_2015_with_labels_v2.csv", nrows = 20000) #517,401 emails, some are labeled with
enron.rename(columns={'content': 'text'}, inplace=True)
enron["score"] = 1
enron["tone"] = "professional"
enron["subreddit"] = "enron"
#%%
"""
Set functions to create dataframes for subreddits of interest.
This is very inefficient right now, SQL would be a better solution.
Inspired by: https://stackoverflow.com/questions/30539679/python-read-several-json-files-from-a-folder
"""
#%%
# bash command to split giant reddit file
# gsplit -d -l 2000000 RC_2018-04.json reddit
# grep "fortnite" reddit01.json > reddit01_test.json

def multi_grep(subreddit, path_to_json):
    json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
    os.mkdir(path_to_json+subreddit)
    for i in json_files:
        out_file = "../data/reddit_chunks/"+subreddit+"/"+subreddit+"_"+i
        if not os.path.isfile(out_file):
            os.system("grep "+subreddit+" ../data/reddit_chunks/"+i+" > ../data/reddit_chunks/"+subreddit+"/"+subreddit+"_"+i)
        print(i)
    print("Done. Proceed to jsons_to_df")
    
def jsons_to_df(subredd, path_to_json):
    json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
    jsons_data = pd.DataFrame(columns=['text', 'subreddit'])
    for i in json_files:
        json_path = path_to_json+i
        print(json_path)
        json_df = pd.read_json(json_path, lines = True)
        json_df = json_df[json_df.subreddit == subredd]
        json_df = json_df[['body', 'subreddit']] #drop unneeded columns
        json_df = json_df.rename(index=str, columns={"body": "text"})
        jsons_data = pd.concat([jsons_data, json_df], ignore_index=True)
    return jsons_data        
#%%
"""
Run Reddit functions on all subreddits I think might be helpful.
"""  
#%%
#multi_grep("fortnite","../data/reddit_chunks/")
#fortnite = jsons_to_df("FORTnITE", "../data/reddit_chunks/FORTnITE/")
subs_to_include = ["datascience","legaladvice", "pics"]
#for i in subs_to_include:
#    multi_grep(i, "../data/reddit_chunks/")

datasci = jsons_to_df("datascience", "../data/reddit_chunks/datascience/")
legadv = jsons_to_df("legaladvice", "../data/reddit_chunks/legaladvice/")
pics = jsons_to_df("pics", "../data/reddit_chunks/pics/")

datasci["score"] = 1
datasci["tone"] = "professional"

legadv["score"] = 1
legadv["tone"] = "professional"

pics["score"] = 0
pics["tone"] = "unprofessional"
pics = pics.iloc[0:20000,]

#%%
"""
Clean the data.
"""
#%%
sources = [hp, ah, datasci, legadv, pics, enron]
subs_to_include = subs_to_include + ["enron", "harrypotter", "askhistorians"]
#all_data = pd.concat([ah.loc[1:8000,["text","subreddit","tone", "score"]], 
#                      hp.loc[1:8000,["text","subreddit","tone", "score"]],
#                      enron.loc[:,["text","subreddit","tone", "score"]]],
#                    sort=False, 
#                    ignore_index=True)
all_data = pd.concat(sources, sort = False, ignore_index=True)
all_data = all_data[all_data["subreddit"].isin(subs_to_include)] #cleaning a few weird rows from enron
all_data["text"] = all_data["text"].fillna(' ') #so that the TfidfVectorizer will work


# function to remove URLs from the data. May add more bits.
# adapted from https://github.com/hundredblocks/concrete_NLP_tutorial/blob/master/NLP_notebook.ipynb
def standardize_text(df, text_field):
    df[text_field] = df[text_field].str.replace(r"http\S+", "")
    df[text_field] = df[text_field].str.replace(r"http", "")
    return df

all_data["text"] = standardize_text(all_data, "text")

all_data["subreddit"].value_counts()
all_data["tone"].value_counts()


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
"""
Testing TF-IDF importance - what's driving the classification? Do we need custom stopwords?
Code adapted from the Insight blog.
"""
#%%
def tfidf(data):
    tfidf_vectorizer = TfidfVectorizer(ngram_range = (1, 2), lowercase = False, stop_words = 'english')
    train = tfidf_vectorizer.fit_transform(data)
    return train, tfidf_vectorizer

def get_most_important_features(vectorizer, model, n=5):
    index_to_word = {v:k for k,v in vectorizer.vocabulary_.items()}
    
    # loop for each class
    classes ={}
    for class_index in range(model.coef_.shape[0]):
        word_importances = [(el, index_to_word[i]) for i,el in enumerate(model.coef_[class_index])]
        sorted_coeff = sorted(word_importances, key = lambda x : x[0], reverse=True)
        tops = sorted(sorted_coeff[:n], key = lambda x : x[0])
        bottom = sorted_coeff[-n:]
        classes[class_index] = {
            'tops':tops,
            'bottom':bottom
        }
    return classes

X_train_tfidf, tfidf_vectorizer = tfidf(X_train)

clf_tfidf = LogisticRegression(C=30.0, class_weight='balanced', solver='newton-cg', 
                         multi_class='multinomial', n_jobs=-1, random_state=40)
clf_tfidf.fit(X_train_tfidf, y_train)

importance_tfidf = get_most_important_features(tfidf_vectorizer, clf_tfidf, 20)
#%%
"""
Custom stopword sets!
"""
#%%
#def tfidf_by_sub(vectorizer, text):
    
def get_top_n_words(corpus, n=None):
    """
    List the top n words in a vocabulary according to occurrence in a text corpus.
    Adapted from https://medium.com/@cristhianboujon/how-to-list-the-most-common-words-from-text-corpus-using-scikit-learn-dad4d0cab41d
    """
    vec = TfidfVectorizer(ngram_range = (1, 2), lowercase = False, stop_words = 'english').fit(corpus)
    tfidf_vec = vec.transform(corpus)
    sum_words = tfidf_vec.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in     vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
    
custom_stopwords = {}

for i in subs_to_include:
    test_input = all_data.loc[all_data["subreddit"]==i,["text"]]
    test_input = test_input.values.tolist()
    test_input = [item for sublist in test_input for item in sublist]
    print("working on "+i)
    common_words = get_top_n_words(test_input, 20)
    custom_stopwords[i] = common_words
    
from nltk.corpus import stopwords

my_stopwords = []
for i in subs_to_include:
    stops = [lis[0] for lis in custom_stopwords[i]]
    my_stopwords = my_stopwords + stops
    
my_stopwords = my_stopwords + stopwords.words('english')
#%%    
    
    