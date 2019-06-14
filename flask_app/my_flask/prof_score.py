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
	logreg = LogisticRegression(n_jobs=1, C=1e5)
	
	# read data and make it a list.
	user_text = [user_text]
	  
	# add tags, though they won't be used in the prediction
	user_text = pd.DataFrame(user_text,columns=['text'])
	user_text["tone"] = "unprofessional"

	# tokenize
	user_text_tagged = user_text.apply(
	    lambda r: TaggedDocument(words=tokenize_text(r['text']), tags=[r.tone]), axis=1)

	# create vectorized version
	y_user_text, X_user_text = vec_for_learning(model_dbow, user_text_tagged)

	# predict professionalism using trained model from above
	prof_score = logreg.predict_proba(X_user_text)

	return prof_score[0][0]





