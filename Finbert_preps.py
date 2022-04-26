#Part 4

import csv
from itertools import count
from re import A, S
from unittest.util import sorted_list_difference
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
import string 

string.punctuation 
import matplotlib.pyplot as plt
import numpy as np
from nltk import FreqDist
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer


import torch
import transformers
from   transformers import AutoTokenizer
from   transformers import AutoModelForSequenceClassification

import pandas as pd


import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from pandas_datareader import data
import os
import re

#Download required tools for FinBert

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

#Creat new columns from the scores

data = pd.read_csv("data_TT.csv")
data["positive"] = ""
data["negative"] = ""
data["neutral"] = ""

#Calculate FinBert scores

for i in range (0,len(data)):
    ec=data.loc[i,'transcript_text']
    inputs = tokenizer(ec, padding = True, truncation = True, return_tensors='pt')
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    data.loc[i,'positive']= predictions[:, 0].tolist()
    data.loc[i,'negative']= predictions[:, 1].tolist()
    data.loc[i,'neutral']= predictions[:, 2].tolist()  

#Remove some columns 
#data.rename(columns={"Unnamed: 0": "Index"}, inplace=True)
data.drop('Unnamed: 0.1', inplace=True, axis=1)
data.drop('Unnamed: 0', inplace=True, axis=1)
data.columns

#Create CSV with true and false 

data_T_F =pd.read_csv("/~/downloads/True_False.csv",sep=';')

data["True_False"] = ""
data['True_False'] =data_T_F['True_False']

data.to_csv('data_TTN.csv')

csv.writer( )
