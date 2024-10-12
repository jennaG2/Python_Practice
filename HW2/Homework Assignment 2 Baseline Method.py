#!/usr/bin/env python
# coding: utf-8

# In[1]:


import scipy
import sklearn
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from collections import Counter


# # Download files, set up folder, put files into folder

# In[2]:


# training data: ./train.tsv
# test data:     ./test.tsv


# # Load training and test data

# In[3]:


dataframe = pd.read_csv('./train.tsv', sep = '\t')
print(dataframe)


# In[4]:


train_ratio = 0.8 # 80% for training, 20% for validation
random_seed = 100

train_dataframe = dataframe.sample(frac=train_ratio, random_state=random_seed)
valid_dataframe = dataframe.drop(train_dataframe.index)
print('training set size:', len(train_dataframe))
print('validation set size:', len(valid_dataframe))


# In[5]:


test_dataframe = pd.read_csv('./test.tsv', sep = '\t')
print (test_dataframe)


# # Try the trivial baseline: predict the majority label of the training set

# In[6]:


Counter(train_dataframe['label'])


# In[7]:


# Looks like label 1 has slightly more counts than label 0 in training data
# So the 'majority guess' prediction is an array filled with 1s
majority_guess_pred = [1 for i in range(len(valid_dataframe))]
accuracy = accuracy_score(valid_dataframe['label'], majority_guess_pred)
print ('Majority guess accuracy:', accuracy)


# In[8]:


# helper function: write out prediction values into a csv format file
# params:
#     df: dataframe, where each row is a test example, with column 'id' as data id
#     pred: a list or 1-d array of prediction values
#     filepath: the output file path
# return:
#     None

def write_test_prediction(df, pred, filepath):
    with open(filepath, 'w') as outfile:
        outfile.write('{},{}\n'.format('id', 'label'))
        for index, row in df.iterrows():
            outfile.write('{},{}\n'.format(row['id'], pred[index]))
    print (len(df), 'predictions are written to', filepath)


# In[9]:


majority_guess_pred_test = [1 for i in range(len(test_dataframe))]
write_test_prediction(test_dataframe, majority_guess_pred_test, './majority_guess.csv')


# # Build feature extractor

# ## use all unigrams from training data as features

# In[71]:


vectorizer = CountVectorizer()
vectorizer.fit(train_dataframe['review'])


# ## Or: reuse the chi-square feature selection method from HW1 

# In[54]:


import re

def process_text(text):
    for punctuations in [',', '.', '"', '!', '?', ':', ';', '-', '(', ')', '[', ']']:
        text = text.replace(punctuations, ' ')
    text = re.sub('\s+', ' ', text)
    text = text.lower().strip()
    return text

def get_single_word_frequency(filepath):
    word_freq = {}
    with open(filepath) as f:
        f.readline() # skip header (the first line) 
        for line in f:
            review_text = process_text(line.split('\t')[1])
            for word in review_text.split():
                if word not in word_freq:
                    word_freq[word] = 1
                else:
                    word_freq[word] += 1
    return word_freq

def get_single_word_doc_frequency_per_label(filepath, label):
    word_freq_per_label = {}
    with open(filepath) as f:
        f.readline() # skip header (the first line) 
        for line in f:
            sentiment_label = line.split('\t')[0].strip()
            if sentiment_label == label:
                review_text = process_text(line.split('\t')[1])
                for word in set(review_text.split()):
                    if word not in word_freq_per_label:
                        word_freq_per_label[word] = 1
                    else:
                        word_freq_per_label[word] += 1
    return word_freq_per_label

def feature_selection_chi2(review_filepath, num_features_to_select):
    num_reviews = 0
    num_positive_reviews = 0
    num_negative_reviews = 0
    with open(review_filepath) as f:
        f.readline() # skip header (the first line) 
        for line in f:
            num_reviews += 1
            if line.strip().split()[0] == '1':
                num_positive_reviews += 1
            else:
                num_negative_reviews += 1
    word_freq = get_single_word_frequency(review_filepath)
    positive_word_freq = get_single_word_doc_frequency_per_label(review_filepath, '1')
    negative_word_freq = get_single_word_doc_frequency_per_label(review_filepath, '0')
    
    chi2_dict = {}
    for word, freq in word_freq.items():
        if word in positive_word_freq and word in negative_word_freq:        
            contingency_table = np.zeros((2,2))
            contingency_table[0][0] = positive_word_freq[word]
            contingency_table[0][1] = negative_word_freq[word]
            contingency_table[1][0] = num_positive_reviews - positive_word_freq[word]
            contingency_table[1][1] = num_negative_reviews - negative_word_freq[word]

            chi2 = 0.0
            for i in range(2):
                for j in range(2):
                    expected_count = sum(contingency_table[i,:])*sum(contingency_table[:,j])/float(num_reviews)
                    chi2 += (contingency_table[i][j] - expected_count)**2 / expected_count

            chi2_dict[word] = chi2
    feature_set = set([])
    for word, chi2 in sorted(chi2_dict.items(), key = lambda x: x[1], reverse = True)[:num_features_to_select]:
        feature_set.add(word)
    return feature_set


# In[60]:


# num_features = 1000
# feature_set = feature_selection_chi2('./train.tsv', num_features)
# vectorizer = CountVectorizer(vocabulary = feature_set)
# vectorizer.fit(train_dataframe['review'])


# # Extract feature vectors for training, validation, and test data 

# In[72]:


train_X = vectorizer.transform(train_dataframe['review'])
valid_X = vectorizer.transform(valid_dataframe['review'])
test_X = vectorizer.transform(test_dataframe['review'])
print (train_X.shape)
print (valid_X.shape)
print (test_X.shape)


# # Train model on training set

# In[73]:


train_Y = train_dataframe['label']
model = LogisticRegression(C = 1, solver='liblinear')
model.fit(train_X, train_Y)


# # Evaluate model on training set

# In[88]:


train_Y_hat = model.predict(train_X)
train_Y = train_dataframe['label'].to_numpy()
accuracy = accuracy_score(train_Y, train_Y_hat)
print ('Logistic regression, accuracy on training set:', accuracy)


# # Evaluate model on validation set

# In[89]:


valid_Y_hat = model.predict(valid_X)
valid_Y = valid_dataframe['label'].to_numpy()
accuracy = accuracy_score(valid_Y, valid_Y_hat)
print ('Logistic regression, accuracy on validation set:', accuracy)


# # After experimentation on the validation set: retrain the final model on all training data, and predict labels for test data

# In[76]:


all_train_Y = dataframe['label']
all_train_X = vectorizer.transform(dataframe['review'])
model.fit(all_train_X, all_train_Y)
test_Y_hat = model.predict(test_X)
write_test_prediction(test_dataframe, test_Y_hat, './logistic_regression.csv')


# In[ ]:





# # Investigate what the model has learned and where it failed (A.K.A. error analysis)

# ## Look at learned parameters (for linear model: weight of each dimension)

# In[77]:


# construct a mapping: word -> learned weight of this word
feature_weight = {}
for word, idx in vectorizer.vocabulary_.items():
    feature_weight[word] = model.coef_[0][idx]


# In[78]:


# words correlated with positive sentiment (top ones)
for k, v in sorted(feature_weight.items(), key = lambda x: x[1], reverse = True)[:10]:
     print (k, v)


# In[79]:


# words correlated with negative sentiments (top ones)
for k, v in sorted(feature_weight.items(), key = lambda x: x[1], reverse = False)[:10]:
     print (k, v)


# ## Look at how the model makes predictions on individual examples

# In[80]:


# We pick a set of examples from the validation set (we predicted scores for those).
# We usually we don't pick from training data (since the good performance may be unrealistic).
# We cannot do error analysis on test data （because no true target value is provided）.


# In[122]:


def explain_linear_prediction(df, model, idx2feature, X, Y, Y_hat, idx_list):
    print('indices:', idx_list)
    for idx in idx_list:
        print ('==============', idx, '================')
        print ('document:', df.iloc[idx]['review'])
        print ('TRUE label:', df.iloc[idx]['label'])
        print ('PRED label:', Y_hat[idx])
        
        print ('\nPRED breakdown:')
        print ('\tINTERCEPT', model.intercept_)
        if X[idx, :].nnz == 0:
            print ('\tFEATURE', '[EMPTY]')
        else:
            sp_row = X[idx, :]
            for i in range(sp_row.getnnz()): # looping over a row in sparse matrix 
                feature_value = sp_row.data[i]
                feature_dim = sp_row.indices[i]
                print ('\tFEATURE', idx2feature[feature_dim], ':', feature_value, '*', model.coef_[0][feature_dim])


# In[123]:


# construct a dictionary mapping: feature index -> word
idx2feature = dict([(v,k) for k,v in vectorizer.vocabulary_.items()])

# look at data with prediction error
error_indices  = [i for i in range(len(valid_Y_hat)) if valid_Y_hat[i] != valid_Y[i]]
explain_linear_prediction(valid_dataframe, model, idx2feature, valid_X, valid_Y, valid_Y_hat, np.random.choice(error_indices, size = 1))

