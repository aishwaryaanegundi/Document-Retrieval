import xml.etree.ElementTree as ElementTree
import re
import collections
import nltk
import string
import math
import operator
import numpy as np

corpus = {}
with open('trec_documents.xml', 'r') as f:   # Reading file
    xml = f.read()

xml = '<ROOT>' + xml + '</ROOT>'   # Let's add a root tag

root = ElementTree.fromstring(xml)

# Simple loop through each document
for doc in root:
    doc_no = doc.find('DOCNO').text.strip()
    text = ''
    if re.search("^LA", doc_no):
        text_element = doc.find('TEXT')
        print(text_element.text)
        for p in doc.find('TEXT').findall('P'):
            text = text + (p.text)
    else:
        text = doc.find('TEXT').text
    corpus[doc_no] = text
processed_corpus = {}
for key, value in corpus.items():
    value = str.lower(value)
    table = str.maketrans('', '', string.punctuation)
    value = value.translate(table)

    processed_corpus[key] = nltk.word_tokenize(value)

#processed_corpus = dict(list(processed_corpus.items())[:int(len(processed_corpus)/2)])
processed_corpus = dict(list(processed_corpus.items())[:1000])
#for key, value in processed_corpus. items():
#    print(key, ' : ', value)

# computing the inverse document frequency
N = len(processed_corpus)
vocabulary = []
print("The number of documents in the corpus: ", N)
idf = {}
doc_count = {}
for key, value in processed_corpus.items():
    for term in value:
        if term not in vocabulary:
            vocabulary.append(term)

for word in vocabulary:
    idf[word] = 0

for word in vocabulary:
    doc_count[word] = 0

for word in vocabulary:
    for key, value in processed_corpus.items():
        if word in value:
            doc_count[word] = doc_count[word] + 1

for word in vocabulary:
    idf[word] = math.log(N/doc_count[word])

# computing term frequencies
term_frequencies = {}
for docno, text in processed_corpus.items():
    tf = {}
    freq_count = {}
    words = []
    for word in text:
        if word not in words:
            words.append(word)
            freq_count[word] = 1
        else:
            freq_count[word] += 1
    max_frequency_key = max(freq_count, key=lambda key: freq_count[key])
    max_frequency = freq_count[max_frequency_key]
    for word in words:
        tf[word] = freq_count[word]/max_frequency
    term_frequencies[docno] = tf

# computing tf-idf scores
tf_idf = {}
for docno, text in processed_corpus.items():
    tfidf = {}
    words = []
    for word in text:
        if word not in words:
            words.append(word)
            tf = (term_frequencies[docno])[word]
            tfidf[word] = tf * idf[word]
    tf_idf[docno] = tfidf

# computing cosine similarity
def cosine_similarity(doc, query):
    cos_sim = np.dot(doc, query)/(np.linalg.norm(doc)*np.linalg.norm(query))
    return cos_sim

# construct an indexed vocabulary
indexed_vocabulary = {}
i = 0
for word in vocabulary:
    indexed_vocabulary[word] = i
    i = i + 1

# vectroizing documents
def vectorize_doc(docno):
    tokens = processed_corpus[docno]
    vector = np.zeros([1,len(vocabulary)])
    for token in tokens:
        index = indexed_vocabulary[token]
        tfidf = tf_idf[docno][token]
        vector[index] = tfidf
    return vector

# vectorize query
def vectorize_query(tokens):
    tf = {}
    freq_count = {}
    words = []
    for word in text:
        if word not in words:
            words.append(word)
            freq_count[word] = 1
        else:
            freq_count[word] += 1
    max_frequency_key = max(freq_count, key=lambda key: freq_count[key])
    max_frequency = freq_count[max_frequency_key]
    for word in words:
        tf[word] = freq_count[word]/max_frequency
    vector = np.zeros([1,len(vocabulary)])
    for token in tokens:
        index = indexed_vocabulary[token]
        tfidf = tf[token]*idf[token]
        vector[index] = tfidf
    return vector
        
    


















































        
        
        
