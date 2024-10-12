#!/usr/bin/env python
# coding: utf-8

import editdistance

# # Question 3: spell correction using letter n-grams

a_list_filepath = 'Desktop/690-270 HW/enwiktionary.a.list'

a_list = []
with open(a_list_filepath) as f:
    for line in f:
        a_list.append(line.strip())

print ('number of words/phrases in the list:', len(a_list))

def chunk_word_into_letter_ngrams(word, n):
    ngrams = []
    for i in range(len(word)-n+1):
        ngrams.append( word[i : i+n] )
    return set(ngrams)

# You need a function that can calculate the Jaccard similarity for any pair of words
def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0


# You also need a function that calculates the edit distance for any pair of words
# (You can use an external package to calculate edit distance, e.g. the "editdistance" package)

def calculate_edit_distance(word1, word2):
    return editdistance.eval(word1, word2) 




# For each given string, you need to find a list of 10 correctly-spelled words from enwiktionary.a.list
#     that have the _highest_ n-gram Jaccard similarity to the given word
# Different lengths of the n-grams (i.e., different n) will likely produce a different list

def find_similar_words_jaccard(input_word, dictionary, n):
    input_ngrams = chunk_word_into_letter_ngrams(input_word, n)
    similar_words = []
    for word in dictionary:
        word_ngrams = chunk_word_into_letter_ngrams(word, n)
        similarity = jaccard_similarity(input_ngrams, word_ngrams)
        similar_words.append((word, similarity))
    similar_words.sort(key=lambda x: x[1], reverse=True)
    return similar_words[:10]   

# For each given string, you need to find a list of 10 correctly-spelled words from enwiktionary.a.list
#     that have the _lowest_ edit distance to the given word
def find_similar_words_edit_distance(input_word, dictionary):
    similar_words = []
    for word in dictionary:
        distance = calculate_edit_distance(input_word, word)
        similar_words.append((word, distance))
    similar_words.sort(key=lambda x: x[1])
    return similar_words[:10]

# testing the function
input_strings = [
"abreviation",
"abstrictiveness",
"accanthopterigious",
"artifitial inteligwnse",
"agglumetation"
]
for input_str in input_strings:
    print("Input String:", input_str)
    print("Top 10 similar words using Jaccard similarity (tri-grams):")
    print(find_similar_words_jaccard(input_str, a_list, 3))
    print("Top 10 similar words using edit distance:")
    print(find_similar_words_edit_distance(input_str, a_list))
    print()

print('BIGRAMS TEST')
#Bigrams
for input_str in input_strings:
    print("Input String:", input_str)
    print("Top 10 similar words using Jaccard similarity (Bi-grams):")
    print(find_similar_words_jaccard(input_str, a_list, 2))
    print("Top 10 similar words using edit distance:")
    print(find_similar_words_edit_distance(input_str, a_list))
    print()

print('4-GRAMS TEST')
#4-grams
for input_str in input_strings:
    print("Input String:", input_str)
    print("Top 10 similar words using Jaccard similarity (4-grams):")
    print(find_similar_words_jaccard(input_str, a_list, 4))
    print("Top 10 similar words using edit distance:")
    print(find_similar_words_edit_distance(input_str, a_list))
    print()    


print('5-GRAMS TEST')    
#5-grams
for input_str in input_strings:
    print("Input String:", input_str)
    print("Top 10 similar words using Jaccard similarity (5-grams):")
    print(find_similar_words_jaccard(input_str, a_list, 5))
    print("Top 10 similar words using edit distance:")
    print(find_similar_words_edit_distance(input_str, a_list))
    print()                