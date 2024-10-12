import scipy
import sklearn
import json
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score
import nltk
nltk.download('punkt')

file_path = 'Desktop/INLS690/Combined_News_DJIA.csv'

dataframe = pd.read_csv(file_path,  keep_default_na = False)
dataframe.head()

len(dataframe)

observations_each_class = dataframe.groupby('Label').size()
print(observations_each_class)

dataframe['Date'] = pd.to_datetime(dataframe['Date'])
dataframe['Year'] = dataframe['Date'].dt.year

# Count the number of observations per year
observations_per_year = dataframe.groupby('Year').size()

print(observations_per_year)

# This selects columns from the 3rd to the 27th
selected_columns = dataframe.iloc[:, 2:27]

# Concatenate the texts in each row
dataframe['combined_row_texts'] = selected_columns.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

# Example outout of the concatenated text for 1 instance
dataframe.iloc[0,28]

# Split the training and testing set
# Data before 1/1/2015 are training, after are testing (aprroximate 70-30 split)
train_df = dataframe[dataframe['Date'] < '2015-01-01']
test_df = dataframe[dataframe['Date'] > '2014-12-31']
print(len(train_df))
print(len(test_df))

train_X = train_df.iloc[:,28]
train_Y = train_df['Label']
test_X = test_df.iloc[:,28]
test_Y = test_df['Label']

# You don't have to use this version of cleaned texts
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def preprocess_text(text):
    # Remove leading "b" characters
    text = re.sub(r'(^|\s)b\'', r'\1', text)  # Handle 'b' followed by a single quote
    text = re.sub(r'(^|\s)b\"', r'\1', text)  # Handle "b" followed by a double quote
    # Lowercasing
    text = text.lower()
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    text = nltk.word_tokenize(text)


    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in text if word not in stopwords.words('english')] # Remove stop-words
    return ' '.join(lemmatized_words)

# Preprocess the data
train_X_cleaned = train_X.apply(preprocess_text)
test_X_cleaned = test_X.apply(preprocess_text)

train_X_cleaned[0]


# Looks like label 1 has slightly more counts than label 0 in training data
# So the 'majority guess' prediction is an array filled with 1s
majority_guess_pred = [1 for i in range(len(test_X_cleaned))]
accuracy = accuracy_score(test_Y, majority_guess_pred)
print('Majority guess accuracy:', accuracy)

print('HELLLOOOOO')