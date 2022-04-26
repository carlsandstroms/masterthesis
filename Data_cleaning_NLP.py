#Part 3

## Clean the data before NLP 
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

#Remove punctuation 
data=pd.read_csv('data_T.csv')
for i in range (0,len(data)):
    Data=data.loc[i,'transcript_text']
    def remove_punctuation(text):
        no_punct=[words for words in text if words not in string.punctuation]
        words_wo_punct=''.join(no_punct)
        return (words_wo_punct)
    Data1=remove_punctuation(Data)

#Remove stop words
            
    stop_words = set(stopwords.words("english"))
    def remove_stopwords(text):
        text1=word_tokenize(text)
        filtered_list = []
        for word in text1:
            if word.casefold() not in stop_words:
                filtered_list.append(word)
        return(filtered_list)
    Data2=remove_stopwords(Data1)    

# Stemming (affixes to suffixes and prefixes)
    def stemming(text):
        text=TreebankWordDetokenizer().detokenize(text)
        lemmatizer = WordNetLemmatizer()
        list=[]
        for token in word_tokenize(text):
            list.append(lemmatizer.lemmatize(token))
        return ' '.join(list)
    Data3=stemming(Data2)
    data.loc[i,'transcript_text']=Data3  

# Store as a new CSV
data.to_csv('data_TT.csv')