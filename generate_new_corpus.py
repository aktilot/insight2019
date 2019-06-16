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