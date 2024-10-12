#!/usr/bin/env python
# coding: utf-8



import re
import scipy
import sklearn
import json
import pandas as pd
import numpy as np
from collections import Counter
from numpy import random
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# # Download files, set up folder, put files into folder

# In[2]:


training_data_path = './reference_metadata_2013.csv'
test_data_path = './reference_metadata_2020.csv'


# In[3]:


# specify data type for each column (to be used in pandas read_csv function)
dtype_dict = {'REFERENCE_ID': str, 'TITLE': str, 'AUTHOR': str, 'YEAR': str, 'ABSTRACT': str, 'CITED': int}


# In[4]:


dataframe = pd.read_csv(training_data_path, dtype = dtype_dict, keep_default_na = False)
dataframe


# In[5]:


train_ratio = 0.7 # 70% for training, 30% for validation
random_seed = 100

train_dataframe = dataframe.sample(frac=train_ratio, random_state=random_seed)
valid_dataframe = dataframe.drop(train_dataframe.index)
print('training set size:', len(train_dataframe))
print('validation set size:', len(valid_dataframe))


# In[6]:


test_dataframe = pd.read_csv(test_data_path, dtype = dtype_dict, keep_default_na = False)
test_dataframe


# Data Preprocessing (before vectorization)
# Text Cleaning
def clean_text(text):
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    # Join tokens back into text
    cleaned_text = ' '.join(filtered_tokens)
    return cleaned_text
# Apply text cleaning to 'TITLE' and 'ABSTRACT' columns
dataframe['TITLE'] = dataframe['TITLE'].apply(clean_text)
dataframe['ABSTRACT'] = dataframe['ABSTRACT'].apply(clean_text)

# Feature Engineering
# Explore additional text features or domain-specific features
# For example, you can extract features related to publication journals, keywords, or citation counts.

# Build feature extractor
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.95, min_df=2)
vectorizer.fit(train_dataframe['TITLE'] + " " + train_dataframe['ABSTRACT'])

# Extract feature vectors for training, validation, and test data
train_X = vectorizer.transform(train_dataframe['TITLE'] + " " + train_dataframe['ABSTRACT'])
valid_X = vectorizer.transform(valid_dataframe['TITLE'] + " " + valid_dataframe['ABSTRACT'])
test_X = vectorizer.transform(test_dataframe['TITLE'] + " " + test_dataframe['ABSTRACT'])

# Train model on training set
train_Y = train_dataframe['CITED']
model = LogisticRegression(C=1, solver='liblinear')
model.fit(train_X, train_Y)

# Evaluate model on training set
train_Y_hat = model.predict_proba(train_X)
train_Y = train_dataframe['CITED'].to_numpy()
wss95 = WSS_95(train_Y, train_Y_hat[:, 1])
print('Logistic regression, WSS@95% on training set:', wss95)

# Evaluate model on validation set
valid_Y_hat = model.predict_proba(valid_X)
valid_Y = valid_dataframe['CITED'].to_numpy()
wss95 = WSS_95(valid_Y, valid_Y_hat[:, 1])
print('Logistic regression, WSS@95% on validation set:', wss95)

# After experimentation on the validation set: retrain the final model on all training data, and predict scores for test data
all_train_Y = dataframe['CITED']
all_train_X = vectorizer.transform(dataframe['TITLE'] + ' ' + dataframe['ABSTRACT'])
model.fit(all_train_X, all_train_Y)
test_Y_hat = model.predict_proba(test_X)

# Write test predictions to a CSV file
def write_test_prediction(df, pred, filepath):
    with open(filepath, 'w') as outfile:
        outfile.write('{},{}\n'.format('REFERENCE_ID', 'Score'))
        for index, row in df.iterrows():
            outfile.write('{},{}\n'.format(row['REFERENCE_ID'], pred[index]))
    print(len(df), 'predictions are written to', filepath)

write_test_prediction(test_dataframe, test_Y_hat[:, 1], './logistic_regression-tfidf-trimmed-bigrams.csv')

# Work saved over sampling at 95% recall (WSS@95%)
def WSS_95(y_true, y_pred):
    res = pd.concat([pd.Series(y_pred), pd.Series(y_true)], axis=1)
    res.columns = ['y_pred', 'y_true']

    # sort res by scores in the submission column
    res.sort_values("y_pred", axis=0, ascending=False, inplace=True)

    # calculate total number of relevant items
    total_num_relevant = sum(res['y_true']) + 1e-100

    # compute recall at each rank until it first surpasses 95%
    curr_num_relevant = 0
    curr_position = 0
    for i, row in res.iterrows():
        curr_num_relevant += row['y_true']
        curr_position += 1
        curr_recall = curr_num_relevant / total_num_relevant
        if curr_recall >= 0.95:
            break

    print('total_num_relevant', total_num_relevant)
    print('curr_position', curr_position)

    WSS_95 = 0.95 - curr_position / len(res)

    return WSS_95





