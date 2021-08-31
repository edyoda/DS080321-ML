# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

dataset = pd.read_csv("Tweets.csv")
print(dataset.isnull().values.any())
print(dataset.isnull().sum(axis=0))

dataset.fillna(0)

# Data Analysis
#Number of tweets for each airline
dataset.airline.value_counts().plot(kind='pie', autopct = '%1.0f%%')

#Sentiment distribution for each airline
dataset_sentiment = dataset.groupby(['airline', 'airline_sentiment']).airline_sentiment.count().unstack()
dataset_sentiment.plot(kind='bar')


#Sentiment distribution
dataset.airline_sentiment.value_counts().plot(kind='pie', autopct = '%1.0f%%')

#Data Cleaning
feature = dataset.iloc[:, 10].values
labels = dataset.iloc[:, 1].values

import re
process_feature = []

for tweet in range(0, len(feature)):
    #1. filtering the special characters \W
    clean_tweet = re.sub(r'\W', ' ', str(feature[tweet]))
    
    #remove numbers
    clean_tweet = re.sub(r'[0-9]+', ' ', clean_tweet)
    
    #2. filtering out single character \s+[a-zA-Z]\s+
    clean_tweet = re.sub(r'\s+[a-zA-Z]\s+', ' ', clean_tweet)
    
    #3. filering out single character from starting ^[a-zA-Z]\s+
    clean_tweet = re.sub(r'^[a-zA-Z]\s+', ' ', clean_tweet)
    
    #4. replace multiple spaces with single space\s+
    clean_tweet = re.sub(r'\s+', ' ', clean_tweet)
        
    #5. Convert to lower case
    clean_tweet = clean_tweet.lower()
    
    process_feature.append(clean_tweet)
    
    
    
