#!/usr/bin/env python3
#%%
import os, re, time
import pandas as pd
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='bs4')

#%%
"""
Here we're loading in the datasets and labeling them.
Harry Potter was acting really differently from the other datasets, so I want to try taking it out.
"""
#%%
# ah # 8,835 comments
ah = pd.read_csv("../data/learning_askhistorians.csv", header = None)
ah = ah.drop([0], axis = 1)
ah.columns = ["text","id","subreddit","meta","time","author","ups","downs","authorlinkkarma","authorkarma","authorisgold"]
ah = ah.dropna(subset = ["text"])
ah["score"] = 1
ah["tone"] = "professional"

# first 8,000 rows of the enron dataset
enron = pd.read_csv("../data/enron_05_17_2015_with_labels_v2.csv", nrows = 50000) #517,401 emails, some are labeled with
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
    for i in json_files:
        out_file = "../data/reddit_chunks/"+subreddit+"/"+subreddit+"_"+i
        if not os.path.isfile(out_file):
            print("Grepping comments from r/"+subreddit)
            os.mkdir(path_to_json+subreddit)
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
subs_to_include = ["datascience", "pics", "FORTnITE"]
#for i in subs_to_include:
#    multi_grep(i, ".../data/reddit_chunks/")

datasci = jsons_to_df("datascience", "../data/reddit_chunks/datascience/")
#legadv = jsons_to_df("legaladvice", "../data/reddit_chunks/legaladvice/")
pics = jsons_to_df("pics", "../data/reddit_chunks/pics/")
fortnite = jsons_to_df("FORTnITE", "../data/reddit_chunks/FORTnITE/")

datasci["score"] = 1
datasci["tone"] = "professional"

#legadv["score"] = 1
#legadv["tone"] = "professional"
#legadv = legadv.iloc[0:20000,]

pics["score"] = 0
pics["tone"] = "unprofessional"
pics = pics.iloc[0:40000,]

fortnite["score"] = 0
fortnite["tone"] = "unprofessional"
#%%
"""
Clean the data.
"""
#%%
sources = [ah, datasci, pics, fortnite, enron]
subs_to_include = subs_to_include + ["enron", "askhistorians"]

all_data = pd.concat(sources, sort = False, ignore_index=True)
all_data = all_data[all_data["subreddit"].isin(subs_to_include)] #cleaning a few weird rows from enron
all_data = all_data[["text","subreddit","tone", "score"]]

# functions to remove URLs from the data. May add more bits.
# adapted from https://github.com/hundredblocks/concrete_NLP_tutorial/blob/master/NLP_notebook.ipynb
# and https://www.kdnuggets.com/2018/03/text-data-preprocessing-walkthrough-python.html

def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def remove_enron_weirdness(text):
    return re.sub('----------------------', '', text)
    return re.sub('Forwarded by', '', text)
    return re.sub('-----Original Message-----', '', text)


def denoise_text(text_str):
    text_str = strip_html(text_str)
    text_str = remove_enron_weirdness(text_str)
    return text_str

in_text = [str(i) for i in all_data["text"]] 
out_text = [denoise_text(x) for x in in_text]
all_data["text"] = out_text
all_data["text"] = all_data["text"].fillna(' ') #so that the TfidfVectorizer will work

## check the balance of the datasets, and adjust previous pieces if unbalanced.
all_data["subreddit"].value_counts()
all_data["tone"].value_counts()

#%%
"""
Happy? Save the output
"""
#%%
all_data.to_csv("../data/190613_corpus.csv")
#%%