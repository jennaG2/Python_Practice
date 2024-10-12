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

train_X


# You don't have to use this version of cleaned texts
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

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

train_X_cleaned_lag = train_X_cleaned.iloc[1:]
train_Y_lag = train_Y.iloc[:-1]
test_X_cleaned_lag = test_X_cleaned.iloc[1:]
test_Y_lag = test_Y.iloc[:-1]

vectorizer = CountVectorizer()


#NAIVE BAYES


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Create a pipeline for vectorizing text and training Naive Bayes classifier
model = make_pipeline(CountVectorizer(), MultinomialNB())

# Train the model
model.fit(train_X_cleaned_lag, train_Y_lag)

# Get the top 20 features/words
vectorizer = model.named_steps['countvectorizer']
classifier = model.named_steps['multinomialnb']
feature_names = vectorizer.get_feature_names_out()
log_prob = classifier.feature_log_prob_  # Log probabilities of features
top_20_index = np.argsort(log_prob[1] - log_prob[0])[-20:]  # Difference of log probabilities for class 1 (Label=1)
top_20_features = np.array(feature_names)[top_20_index]

print("Top 20 features/words:")
print(top_20_features)

# Predict on the test set
predicted = model.predict(test_X_cleaned_lag)

# Find an example of an incorrect prediction
incorrect_index = np.where(predicted != test_Y_lag)[0][0]
incorrect_text = test_X_cleaned_lag.iloc[incorrect_index]
correct_label = test_Y_lag.iloc[incorrect_index]
predicted_label = predicted[incorrect_index]

print("\nExample of an incorrect prediction:")
print("Correct label:", correct_label)
print("Predicted label:", predicted_label)
print("Text:")
print(incorrect_text)

from sklearn.metrics import accuracy_score

# Predict on the test set
predicted = model.predict(test_X_cleaned_lag)

# Calculate accuracy
accuracy = accuracy_score(test_Y_lag, predicted)
print("Accuracy:", accuracy)
