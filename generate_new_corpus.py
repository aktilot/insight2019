#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 20:31:34 2019

@author: amandakomuro
"""
#%%
import csv
import glob
from bs4 import BeautifulSoup
import lxml
import os, re
import pandas as pd
#%%
"""
Convert dissertations and govt repoprts to .csv, each line of original PDF is a line in the file.
"""
#%%
import tika
tika.initVM()
from tika import parser

#with open('./data/Psychology_dissertation_1.csv', 'w') as csvFile:
#    writer = csv.writer(csvFile)
#    writer.writerow([parsed["content"]])
#
#csvFile.close()


dissertations = [i for i in glob.glob("./data/Dissertations/*.pdf")]

for i in range(0,len(dissertations)):
    parsed = parser.from_file(dissertations[i])
    with open("./data/Dissertations/dissertation_"+str(i)+".csv", 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow([parsed["content"]])
    csvFile.close()
    
reports = [i for i in glob.glob("./data/GovernmentReports/*.pdf")]

for i in range(0,len(reports)):
    parsed = parser.from_file(reports[i])
    with open("./data/GovernmentReports/report_"+str(i)+".csv", 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow([parsed["content"]])
    csvFile.close()  
    

#%%
"""
Turn docs into lists, 1 page per item?
Note: I manually removed the table of contents, tables, and references, leaving just the 
main body text.
"""
#%%
all_diss_files = glob.glob("./data/Dissertations/dissertation_*_clean1.csv")
all_diss_pages = []

for i in all_diss_files:
    diss_pages = []
    page_builder = []
    with open(i) as csvFile:
        readCSV = csv.reader(csvFile)
        for row in readCSV:
            row = ''.join(row) #makes row into a string, not list
            page_builder.append(row) 
            temp_row = row.strip()
            if temp_row.isdigit(): #marks a page break in the original pdf
                print("found page break")
                diss_pages.append(page_builder)
                page_builder = []
                continue
    all_diss_pages.extend(diss_pages) 

all_diss_pages2 = [''.join(i) for i in all_diss_pages]

#%%
"""
Clean up Ask a Manager text:
    These are posts from 2017-2019, pulled using the Wayback Machine
"""
#%%
def striphtml(data):
    p = re.compile(r'<.*?>')
    return p.sub('', data)

def clean_post_ends(data):
    p = re.compile(r'You may also like:.*$')
    return p.sub('', data)

def clean_punct(data):
    p = re.compile(r'&#8217;')
    q = re.compile(r'&#8220;')
    r = re.compile(r'&#8221;')
    s = re.compile(r'&#8212;')
    corrected = p.sub('\'', data)   
    corrected = q.sub('“', corrected) 
    corrected = r.sub('”', corrected) 
    corrected = s.sub('—', corrected)             
    return corrected

def process_AskAManager(aam_xml_file): 
    aam_text = []
    with open(aam_xml_file,"r") as aam_xml:
        contents = aam_xml.read()
        soup = BeautifulSoup(contents, "xml")
        posts = soup.find_all("blog_content") # This is a list of everything within this tag
        for post in posts:
            text = post.contents[0]
            aam_text.append(text)
    aam_text = [striphtml(post) for post in aam_text]
    aam_text = [clean_post_ends(post) for post in aam_text]
    aam_text = [clean_punct(post) for post in aam_text]
    return aam_text

aam = glob.glob("./data/AskAManager/*.xml")
for i in range(0,len(aam)):
    os.system("sed 's/content:encoded/blog_content/g' "+aam[i]+" > ./data/AskAManager/aam_clean"+str(i)+".xml")

clean_aam = glob.glob("./data/AskAManager/aam_clean*.xml")
all_ask_a_manager = []
for i in clean_aam:
    this_aam_text = process_AskAManager(i)
    all_ask_a_manager.extend(this_aam_text) 
    
#%%
"""
Clean up Public Sector Consulting reports
These are downloaded from their website.
Note: I manually removed tables first.
"""
#%%
all_gov_files = glob.glob("./data/GovernmentReports/report_*_clean1.csv")
all_gov_pages = []

for i in all_gov_files:
    report_pages = []
    page_builder = []
    with open(i) as csvFile:
        readCSV = csv.reader(csvFile)
        for row in readCSV:
            row = ''.join(row) #makes row into a string, not list
            page_builder.append(row) 
            temp_row = row.strip()
            if temp_row.isdigit(): #marks a page break in the original pdf
                print("found page break")
                report_pages.append(page_builder)
                page_builder = []
                continue
    all_gov_pages.extend(report_pages) 

all_gov_pages2 = [''.join(i) for i in all_gov_pages]

#%%
"""
Get sub reddits for "very casual internet speak" and "professionals discussing 
their work informally"
"""
#%%
def multi_grep(subreddit, path_to_json):
    json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
    if not os.path.isfile("./data/reddit_chunks/"+subreddit+"/"+subreddit+"_"+json_files[0]):
        print("Grepping comments from r/"+subreddit)
        os.mkdir(path_to_json+subreddit)
        for i in json_files:
            out_file = "./data/reddit_chunks/"+subreddit+"/"+subreddit+"_"+i
            os.system("grep "+subreddit+" ./data/reddit_chunks/"+i+" > "+out_file)
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
        jsons_data["source"] = "r_"+subredd
    return jsons_data  

subs_to_include = ["datascience", "aww", "FORTnITE", "marketing", "publishing", "RenewableEnergy", "AskHistorians"]
for i in subs_to_include:
    multi_grep(i, "./data/reddit_chunks/")
#
df_datascience = jsons_to_df("datascience", "./data/reddit_chunks/datascience/")
df_datascience["source"] = "Slack-like"
df_FORTnITE = jsons_to_df("FORTnITE", "./data/reddit_chunks/FORTnITE/")
df_FORTnITE["source"] = "Extremely Casual"
df_aww = jsons_to_df("aww", "./data/reddit_chunks/aww/")
df_aww["source"] = "Extremely Casual"
df_marketing = jsons_to_df("marketing", "./data/reddit_chunks/marketing/")
df_marketing["source"] = "Slack-like"
df_publishing = jsons_to_df("publishing", "./data/reddit_chunks/publishing/")
df_publishing["source"] = "Slack-like"
df_RenewableEnergy = jsons_to_df("RenewableEnergy", "./data/reddit_chunks/RenewableEnergy/")
df_RenewableEnergy["source"] = "Slack-like"
df_AskHistorians = jsons_to_df("AskHistorians", "./data/reddit_chunks/AskHistorians/")
df_AskHistorians["source"] = "Slack-like"

#%%
"""
Bind all of the text sources together!
"""
#%%

df_gov = pd.DataFrame(all_gov_pages2, columns=["text"])
df_gov["source"] = "Governmental"
df_aam = pd.DataFrame(all_ask_a_manager, columns=["text"])
df_aam["source"] = "Workplace_Casual"
df_diss = pd.DataFrame(all_diss_pages2, columns=["text"])
df_diss["source"] = "Dissertation"

new_corpus =  pd.concat([df_datascience,
                        df_FORTnITE.iloc[0:5000,],
                        df_aww.iloc[0:5000,],
                        df_marketing,
                        df_publishing,
                        df_RenewableEnergy,
                        df_AskHistorians,
                        df_gov, 
                        df_aam, 
                        df_diss], 
                        sort = False, 
                        ignore_index=True)

## check the balance of the datasets, and adjust previous pieces if unbalanced.
new_corpus["source"].value_counts()


#%%
"""
Saving the output, classes are not currently balanced.
Might consider splitting the Dissertations and Governmental pages 
into smaller strings.
Slack-like          16573
Extremely Casual    10000
Workplace_Casual     1215
Dissertation          423
Governmental          215
"""
#%%
new_corpus.to_csv("./data/190615_corpus.csv")
