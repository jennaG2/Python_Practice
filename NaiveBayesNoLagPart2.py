import nltk
import scipy
import sklearn
import json
import pandas as pd
import numpy as np
import nltk
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

file_path = 'Desktop/INLS690/Combined_News_DJIA.csv'

dataframe = pd.read_csv(file_path,  keep_default_na=False)

observations_each_class = dataframe.groupby('Label').size()
print(observations_each_class)

dataframe['Date'] = pd.to_datetime(dataframe['Date'])
dataframe['Year'] = dataframe['Date'].dt.year

observations_per_year = dataframe.groupby('Year').size()
print(observations_per_year)

selected_columns = dataframe.iloc[:, 2:27]

dataframe['combined_row_texts'] = selected_columns.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

train_df = dataframe[dataframe['Date'] < '2015-01-01']
test_df = dataframe[dataframe['Date'] > '2014-12-31']
print(len(train_df))
print(len(test_df))

train_X = train_df.iloc[:, 28]
train_Y = train_df['Label']
test_X = test_df.iloc[:, 28]
test_Y = test_df['Label']

train_X

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def preprocess_text(text):
    text = re.sub(r'(^|\s)b\'', r'\1', text)
    text = re.sub(r'(^|\s)b\"', r'\1', text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    text = nltk.word_tokenize(text)

    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in text if word not in stopwords.words('english')]
    return ' '.join(lemmatized_words)

train_X_cleaned = train_X.apply(preprocess_text)
test_X_cleaned = test_X.apply(preprocess_text)

vectorizer = CountVectorizer()

model = make_pipeline(vectorizer, MultinomialNB())

model.fit(train_X_cleaned, train_Y)

# Get the top 20 features/words
feature_names = vectorizer.get_feature_names_out()
log_prob = model.named_steps['multinomialnb'].feature_log_prob_
top_20_index = np.argsort(log_prob[1] - log_prob[0])[-20:]
top_20_features = np.array(feature_names)[top_20_index]

print("Top 20 features/words:")
print(top_20_features)

predicted = model.predict(test_X_cleaned)

# Find an example of an incorrect prediction
incorrect_index = np.where(predicted != test_Y)[0][0]
incorrect_text = test_X_cleaned.iloc[incorrect_index]
correct_label = test_Y.iloc[incorrect_index]
predicted_label = predicted[incorrect_index]

print("\nExample of an incorrect prediction:")
print("Correct label:", correct_label)
print("Predicted label:", predicted_label)
print("Text:")
print(incorrect_text)

accuracy = accuracy_score(test_Y, predicted)
print("Accuracy:", accuracy)
