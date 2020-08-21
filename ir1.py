import xml.etree.ElementTree as ElementTree
import re
import collections
import nltk
import string
import math
import operator
import numpy as np
import collections
from bs4 import BeautifulSoup as bs

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
print("size of vocabulary: ", len(vocabulary))
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
    vector = np.zeros(len(vocabulary))
    for token in tokens:
        index = indexed_vocabulary[token]
        tfidf = tf_idf[docno][token]
        vector[index] = tfidf
#    print(np.nonzero(vector))
    return vector

# vectorize query
def vectorize_query(tokens):
    qtf = {}
    counts = {}
    query_words = []
    for word in tokens:
        if word not in query_words:
            query_words.append(word)
            counts[word] = 1
        else:
            counts[word] += 1

    max_frequency_key = max(counts, key=lambda key: counts[key])
    max_frequency = counts[max_frequency_key]
    for word in query_words:
        qtf[word] = counts[word]/max_frequency
    vector = np.zeros(len(vocabulary))
    for token in tokens:
        try:
            index = indexed_vocabulary[token]
            tfidf = qtf[token]*idf[token]
            vector[index] = tfidf
        except KeyError:
            print("token: ", token, " not found")
    return vector

# compute similarity scores
def get_top_n_relevant_doc(query, n):
#query = "Who is the author of the book, The Iron Lady: A Biography of Margaret Thatcher?"
    query = str.lower(query)
    table = str.maketrans('', '', string.punctuation)
    query = query.translate(table)
    query_tokens = nltk.word_tokenize(query)

    query_vector = vectorize_query(query_tokens)
    similarity_scores = {}
    for docno, text in processed_corpus.items():
        doc_vector = vectorize_doc(docno)
        score = cosine_similarity(doc_vector, query_vector)
        similarity_scores[docno] = score

    sorted_scores = sorted(similarity_scores.items(), key=lambda kv: kv[1])
    sorted_similarity_scores = collections.OrderedDict(sorted_scores)

    # return the top 50 relevant documents
    ranked_documents = {}
    i = 0
    for docno, score in reversed(sorted_similarity_scores.items()):
        ranked_documents[i + 1] = [docno, score]
        i = i + 1
        if i == n:
            break
    return ranked_documents
#print(ranked_documents)

# Evaluation
# Extraction of queries
queries = []
content = []
with open("test_questions.txt", "r") as file:
    content = file.readlines()
    content = "".join(content)
    bs_content = bs(content, "lxml")
    result = bs_content.find_all('desc')
    for query in bs_content.find_all('desc'):
        queries.append((query.text).replace('Description:\n', ''))

# Extraction of patterns
patterns = {}
with open("patterns.txt", "r") as file:
    content = file.readlines()
    for line in content:    
        tokens = line.split(' ', 1)
        if tokens[0] not in patterns:
            key = tokens[0]
            pattern = []
            pattern.append(tokens[1].replace('\n', ''))
            patterns[key] = pattern
        else:  
            (patterns[tokens[0]]).append(tokens[1].replace('\n', ''))

#print(patterns)
query_to_pattern_map = {}
i = 1
for query in queries:
    query_to_pattern_map[query] = patterns[str(i)]
    i = i + 1

# Check if the document is relevant
def isRelevant(docno, query):
    text = processed_corpus[docno]
    query_patterns = query_to_pattern_map[query]
    for pattern in query_patterns:
        for token in text:
            if re.match(pattern, token):
                return True
    return False

# Compute mean precision scores
precision_sum = 0
for query in query_to_pattern_map:
    relevant_count = 0
    retrieved_docs = get_top_n_relevant_doc(query, 50)
    for rank, docno in retrieved_docs.items():
        if isRelevant(docno[0], query):
            relevant_count = relevant_count + 1
    precision = relevant_count / 50.0
    precision_sum = precision_sum + precision
mean_precision = precision_sum/100.0
print(mean_precision)        
    















































        
        
        
