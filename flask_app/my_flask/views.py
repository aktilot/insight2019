from flask import Flask, render_template, request
import requests
import pandas as pd
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import csv
import pickle


# from my_flask import app

# importing custom modules (functions for running my model)
#from prof_score import tokenize_text, vec_for_learning, get_professionalism_score


#Initialize app
app = Flask(__name__, static_url_path='/static')


#Standard home page. 'index.html' is the file in your templates that has the CSS and HTML for your app
@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

#Load the pickled objects for my model and other needed parts.
# model_dbow = Doc2Vec.load("./model/canisaythat_d2v.model")
model_dbow = pickle.load(open("./model/finalized_model.sav", 'rb'))
logreg = pickle.load(open("./model/finalized_model2.sav", 'rb'))


my_stopwords = []
with open('./model/custom_stops_plus_nltk.csv', 'r') as f:
    reader = csv.reader(f)
    my_stopwords = list(reader)
my_stopwords = [item for sublist in my_stopwords for item in sublist]


#Putting everything that was in prof_score.py into views.py for now.
import pandas as pd
import nltk
from sklearn.linear_model import LogisticRegression
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text): #sent is for sentence
        for word in nltk.word_tokenize(sent, language='english'): # language='english'
            if word in my_stopwords: #using my custom set of subreddit stopwords plus NLTK's
                continue
            tokens.append(word) #(word.lower()) if I want to make everything lowercase.
    return tokens

def vec_for_learning(model, tagged_docs):
    sents = tagged_docs.values
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, regressors



def get_professionalism_score(user_text): #this is the function for running text through my NLP model
  """
  Pretty sure the full code for running my model needs to go here?
  """
  # logreg = LogisticRegression(n_jobs=1, C=1e5)
      
  # add tags, though they won't be used in the prediction
  user_text = pd.DataFrame(user_text,columns=['text'])
  user_text["tone"] = "unprofessional"

  # tokenize
  user_text_tagged = user_text.apply(lambda r: TaggedDocument(words=tokenize_text(r['text']), tags=[r.tone]), axis=1)

  # create vectorized version
  y_user_text, X_user_text = vec_for_learning(model_dbow, user_text_tagged)

  # predict professionalism using trained model from above
  prof_score = logreg.predict_proba(X_user_text)

  return prof_score[0][0]


#Back to defining views
#My home page redirects to recommender.html where the user fills out a survey (user input)
@app.route('/input', methods=['GET', 'POST'])
def get_inputs():
    return render_template('inputs.html')

#After they submit the survey, the recommender page redirects to recommendations.html
@app.route('/output', methods=['GET', 'POST'])
def get_outputs():

    user_text = str(request.form['user_text'])
    # user_text = "This is a placeholder sentence."
    prof_score = "placeholder"

    # read data (should be a string) and make it a list.
    user_text1 = [i for i in user_text]

    # run the model function
    prof_score = get_professionalism_score(user_text1)
    prof_score = prof_score * 100
    prof_score = int(round(prof_score))


    #arguments are whatever comes out of your app, in my case a cos_sim and the recommended florist
    #the structure is render_template('your_html_file.html', x=x, y=y, etc...)
    #refer to my recommendations.html to see how variables work
    return render_template(
      "outputs.html", 
      values = { 
          'user_text': user_text if user_text else "No input.",
          'prof_score': prof_score if prof_score else "No result."
      })






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