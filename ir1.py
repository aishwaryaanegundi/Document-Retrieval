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
            if re.match(pattern, token, flags = re.IGNORECASE):
                return True
    return False

# Compute mean precision scores
#precision_sum = 0
#for query in query_to_pattern_map:
#    relevant_count = 0
#    retrieved_docs = get_top_n_relevant_doc(query, 50)
#    for rank, docno in retrieved_docs.items():
#        if isRelevant(docno[0], query):
#            print("relevant: ", docno[0], query)
#            relevant_count = relevant_count + 1
#    precision = relevant_count / 50.0
#    print("precision for each query: ", query, precision)
#    precision_sum = precision_sum + precision
#mean_precision = precision_sum/100.0
#print("baseline mean precision: ", mean_precision)        
    

# Task 2
# BM25
# construct inverted index
inverted_index = {}
for docno, text in processed_corpus.items():
    for word in text:
        if word in inverted_index:
            if docno in inverted_index[word]:
                inverted_index[word][docno] += 1
            else:
                inverted_index[word][docno] = 1
        else:
            d = dict()
            d[docno] = 1
            inverted_index[word] = d

# Compute document frequency
def get_document_frequency(word, docno):
    if word in inverted_index:
        return inverted_index[word][docno]
    else:
        raise LookupError('%s not in index' % word)

# construct document length table
document_length = {}
for docno, text in processed_corpus.items():
    document_length[docno] = len(text)

# compute average document length
total_length = 0
for docno, length in document_length.items():
    total_length += length
average_document_length = total_length / float(len(document_length))

# compute bm25 score
k1 = 1.2
k2 = 100
b = 0.75
R = 0.0


def score_BM25(n, f, qf, r, N, dl, avdl):
	K = compute_K(dl, avdl)
	first = math.log( ( (r + 0.5) / (R - r + 0.5) ) / ( (n - r + 0.5) / (N - n - R + r + 0.5)) )
	second = ((k1 + 1) * f) / (K + f)
	third = ((k2+1) * qf) / (k2 + qf)
	return first * second * third


def compute_K(dl, avdl):
	return k1 * ((1-b) + b * (float(dl)/float(avdl)) )

# get top 50 reranked relevant documents
def get_top_n_reranked_relevant_docs(query, baseline_docs, n):
    query = str.lower(query)
    table = str.maketrans('', '', string.punctuation)
    query = query.translate(table)
    query_tokens = nltk.word_tokenize(query)
    bm25_scores = {}
    for qt in query_tokens:
        if qt in inverted_index:
            doc_dict = inverted_index[qt] 
            for docno, freq in doc_dict.items():
                if docno in baseline_docs:
                    score = score_BM25(n=len(doc_dict), f = freq, qf = 1, r = 0, N = len(document_length),
									       dl = document_length[docno], avdl = average_document_length)
                    if docno in bm25_scores:
                        bm25_scores[docno] += score
                    else:
                        bm25_scores[docno] = score
    sorted_scores = sorted(bm25_scores.items(), key=lambda kv: kv[1])
    sorted_bm25_scores = collections.OrderedDict(sorted_scores)

    # return the top 50 relevant documents
    ranked_documents = {}
    i = 0
    for docno, score in reversed(sorted_bm25_scores.items()):
        ranked_documents[i + 1] = [docno, score]
        i = i + 1
        if i == n:
            break
    return ranked_documents

# Compute mean precision scores
precision_sum = 0
for query in query_to_pattern_map:
    baseline_retrieved_docs = []
    baseline_rel_docs = get_top_n_relevant_doc(query, 1000)
    for rank, doc in baseline_rel_docs.items():
        baseline_retrieved_docs.append(doc[0])
    relevant_count = 0
    retrieved_docs = get_top_n_reranked_relevant_docs(query, baseline_retrieved_docs, 50)
    for rank, docno in retrieved_docs.items():
        if isRelevant(docno[0], query):
#            print("relevant: ", docno[0], query)
            relevant_count = relevant_count + 1
    precision = relevant_count / 50.0
    print("precision for each query: ", query, precision)
    precision_sum = precision_sum + precision
mean_precision = precision_sum/100.0
print("BM25 mean precision: ", mean_precision)   







































        
        
        
