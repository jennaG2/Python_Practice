#!/usr/bin/env python
# coding: utf-8

import re
import math

# # Question 1: word association mining

# ## Basic statistics of the corpus



review_filepath = 'Desktop/690-270 HW/amazon_reviews.txt'



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
print ('total number of reviews:', num_reviews)
print ('total number of positive reviews:', num_positive_reviews)
print ('total number of negative reviews:', num_negative_reviews)


# ## Count frequency of single words




def process_text(text):
    for punctuations in [',', '.', '"', '!', '?', ':', ';', '-', '(', ')', '[', ']']:
        text = text.replace(punctuations, ' ')
    text = re.sub('\s+', ' ', text)
    text = text.lower().strip()
    return text





# Count the frequency of single words (aka. unigrams) in the corpus
# Parameter:
#       filepath: file path of amazon_review.txt
# Return: 
#       a dictionary, key = word, value = word frequency

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





word_freq = get_single_word_frequency(review_filepath)
for word, freq in sorted(word_freq.items(), key = lambda x: x[1], reverse = True)[:10]:
    print(word, freq)





total_num_words = sum(word_freq.values())
print ('number of unique words:', len(word_freq))
print ('total number of word occurrences:', total_num_words)


# ## Count frequency of ordered pair of words in a text window




# Count the number of text windows that contain an ordered pair of words
# Parameter:
#       filepath: file path of amazon_review.txt
#       window_size: the size of a text window (measured in number of words)
# Return: 
#       a dictionary, key = ordered word pair (a tuple), 
#                     value = number of text windows containing this pair

def get_ordered_word_pair_frequency(filepath, window_size):
    pair_freq = {}
    with open(filepath) as f:
        f.readline() # skip header (the first line) 
        for line in f:
            review_text = process_text(line.split('\t')[1])
            word_list = review_text.split()
            for i in range(len(word_list)):
                for j in range(i + 1, len(word_list)):
                    # only consider pairs of words no more than window_size apart  
                    if j - i + 1 >= window_size:
                        break
                    # put this ordered word pair into a tuple
                    order_word_pair = (word_list[i], word_list[j])
                    # accumulate counts
                    if order_word_pair not in pair_freq:
                        pair_freq[order_word_pair] = 1
                    else:
                        pair_freq[order_word_pair] += 1
    return pair_freq





TEXT_WINDOW_SIZE = 5
pair_freq = get_ordered_word_pair_frequency(review_filepath, TEXT_WINDOW_SIZE)
for pair, freq in sorted(pair_freq.items(), key = lambda x: x[1], reverse = True)[:10]:
    print(pair, freq)


# ## Calculate pointwise mutual information for each ordered pair

def calculate_pmi_per_pair(pair_freq, word_freq, total_num_words):
    WORD_PAIR_FREQUENCY_THRESHOLD = 50
    pmi_per_pair = {}
    for pair, freq in pair_freq.items():
       if freq < WORD_PAIR_FREQUENCY_THRESHOLD: # filter out infrequent word pairs

           continue
       if pair[0] in word_freq and pair[1] in word_freq:
        pmi = math.log2((freq * total_num_words) / (word_freq[pair[0]] * word_freq[pair[1]]))
        pmi_per_pair[pair] = pmi       
    return pmi_per_pair

pmi_per_pair = calculate_pmi_per_pair(pair_freq, word_freq, total_num_words)

# Sort word pairs by their PMI from highest to lowest
sorted_pmi_pairs = sorted(pmi_per_pair.items(), key=lambda x: x[1], reverse=True)

# Show the top 100 pairs
top_100_pmi_pairs = sorted_pmi_pairs[:100]

# Print the top 100 pairs
for pair, pmi in top_100_pmi_pairs:
    print(pair, pmi)


print ("QUESTION 2")

#######

# # Question 2: feature selection using Chi-square statistic

# ## For each word, count how many positive (negative) documents it appears in




# Count the number of documents that has a specified sentiment and contain a single word  
# Parameter:
#       filepath: file path of amazon_review.txt
#       label: string '0' (negative) or '1' (positive).   
# Return: 
#       a dictionary, key = word, value = word frequency

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





# number of positive documents that contain a word 
positive_word_freq = get_single_word_doc_frequency_per_label(review_filepath, '1')
for word, freq in sorted(positive_word_freq.items(), key = lambda x: x[1], reverse = True)[:10]:
    print(word, freq)





# number of negative documents that contain a word 
negative_word_freq = get_single_word_doc_frequency_per_label(review_filepath, '0')
for word, freq in sorted(negative_word_freq.items(), key = lambda x: x[1], reverse = True)[:10]:
    print(word, freq)


# ## Calculate Chi-square statistic for each word




# contingency table per word:
#                                             sentiment
#                       positive                            negative
#               ------------------------------------------------------------------------
#       present | word present, positive sentiment | word present, negative sentiment |  
# word          ------------------------------------------------------------------------
#       absent  | word absent,  positive sentiment | word absent, negative sentiment  |  
#               ------------------------------------------------------------------------
#      

chi2_per_word = {}
for word, freq in word_freq.items():
     if freq < 10: # filter infrequent words
         continue
     
     if word in positive_word_freq and word in negative_word_freq:   
         present_positive = positive_word_freq[word]
         present_negative = negative_word_freq[word]
         present_total = present_positive + present_negative
         absent_positive = num_positive_reviews - present_positive
         absent_negative = num_negative_reviews - present_negative
         absent_total = absent_positive + absent_negative

        # Calculate expected frequencies
         expected_present_positive = (present_total * (present_positive + absent_positive)) / num_reviews
         expected_present_negative = (present_total * (present_negative + absent_negative)) / num_reviews
         expected_absent_positive = (absent_total * (present_positive + absent_positive)) / num_reviews
         expected_absent_negative = (absent_total * (present_negative + absent_negative)) / num_reviews

        # Calculate Chi-square statistic
         chi2 = ((present_positive - expected_present_positive) ** 2) / expected_present_positive
         chi2 += ((present_negative - expected_present_negative) ** 2) / expected_present_negative
         chi2 += ((absent_positive - expected_absent_positive) ** 2) / expected_absent_positive
         chi2 += ((absent_negative - expected_absent_negative) ** 2) / expected_absent_negative

         chi2_per_word[word] = chi2
       
    
#Sorted words
sorted_chi2_words = sorted(chi2_per_word.items(), key=lambda x: x[1], reverse=True)

# Top 100 words
top_100_chi2_words = sorted_chi2_words[:100]


# Print the top 100 words
for word, chi2 in top_100_chi2_words:
    print(word, chi2)






