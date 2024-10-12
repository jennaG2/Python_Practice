#!/usr/bin/env python
# coding: utf-8

import scipy
import sklearn
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from collections import Counter
from sklearn.model_selection import cross_val_score

# Load training and test data
dataframe = pd.read_csv('./train.tsv', sep='\t')
print(dataframe)

train_ratio = 0.8  # 80% for training, 20% for validation
random_seed = 100

train_dataframe = dataframe.sample(frac=train_ratio, random_state=random_seed)
valid_dataframe = dataframe.drop(train_dataframe.index)
print('training set size:', len(train_dataframe))
print('validation set size:', len(valid_dataframe))

test_dataframe = pd.read_csv('./test.tsv', sep='\t')
print(test_dataframe)

# Remove stopwords
stopwords_list = ["the", "is", "and", "in", "of", "to", "a", "for", "on", "with", "that"]  # Add more stopwords as needed

def remove_stopwords(text):
    tokens = text.split()  # Tokenize the text into words
    filtered_tokens = [token for token in tokens if token.lower() not in stopwords_list]  # Remove stopwords
    return ' '.join(filtered_tokens)  # Join the filtered tokens back into a string


# Build feature extractor
vectorizer = CountVectorizer(ngram_range=(1, 2))
vectorizer.fit(train_dataframe['review'])

# Extract feature vectors for training, validation, and test data 
train_X = vectorizer.transform(train_dataframe['review'])
valid_X = vectorizer.transform(valid_dataframe['review'])
test_X = vectorizer.transform(test_dataframe['review'])
print(train_X.shape)
print(valid_X.shape)
print(test_X.shape)

# Define model
model = LogisticRegression(C=1, solver='liblinear')

# Perform cross-validation
all_X = vectorizer.transform(dataframe['review'])
all_Y = dataframe['label']
cv_scores = cross_val_score(model, all_X, all_Y, cv=5)  # 5-fold cross-validation
print("Cross-validation scores:", cv_scores)
print("Mean cross-validation accuracy:", np.mean(cv_scores))

# Train model on all training data
model.fit(all_X, all_Y)

# Evaluate model on validation set
valid_Y_hat = model.predict(valid_X)
valid_Y = valid_dataframe['label'].to_numpy()
accuracy = accuracy_score(valid_Y, valid_Y_hat)
print('Logistic regression, accuracy on validation set:', accuracy)

# Make predictions on test data
test_Y_hat = model.predict(test_X)

# Write predictions to file
def write_test_prediction(df, pred, filepath):
    with open(filepath, 'w') as outfile:
        outfile.write('{},{}\n'.format('id', 'label'))
        for index, row in df.iterrows():
            outfile.write('{},{}\n'.format(row['id'], pred[index]))
    print(len(df), 'predictions are written to', filepath)

write_test_prediction(test_dataframe, test_Y_hat, './logistic_regression_cv.csv')

# Investigate what the model has learned and where it failed (A.K.A. error analysis)
# Note: Add your error analysis code here if needed



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

# Look at data with prediction error
error_indices = [i for i in range(len(valid_Y_hat)) if valid_Y_hat[i] != valid_Y[i]]
if error_indices:  # Check if error_indices is not empty
    explain_linear_prediction(valid_dataframe, model, idx2feature, valid_X, valid_Y, valid_Y_hat, np.random.choice(error_indices, size=1))
else:
    print("No prediction errors found in the validation set.")