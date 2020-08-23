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
from nltk.tokenize import sent_tokenize

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
#processed_corpus = dict(list(processed_corpus.items())[:1000])

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
print("Size of vocabulary: ", len(vocabulary))
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
            pass
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
    text = corpus[docno]
    query_patterns = query_to_pattern_map[query]
    for pattern in query_patterns:
        if re.search(pattern, text, flags = re.IGNORECASE):
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
print("Baseline mean precision: ", mean_precision)        
    

# Task 2
# BM25
# construct frequency count table
frequencies = {}
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
    for word in words:
        tf[word] = freq_count[word]
    frequencies[docno] = tf
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
b = 0.75

def compute_BM25_score(qt, docno):
    score = 0
    try:
        inverse_doc_freq = math.log(((N - doc_count[qt]+ 0.5)/(doc_count[qt] + 0.5)) + 1)
        f_qi_d = frequencies[docno][qt]
        score = inverse_doc_freq * f_qi_d * (((k1+1)/(f_qi_d + (k1 * (1-b+(b * document_length[docno]/average_document_length))))) + 1)
    except KeyError:
        pass
    return score



def get_top_n_reranked_relevant_docs(query, baseline_docs, n):
    query = str.lower(query)
    table = str.maketrans('', '', string.punctuation)
    query = query.translate(table)
    query_tokens = nltk.word_tokenize(query)
    bm25_scores = {}
    for qt in query_tokens:
        for docno in baseline_docs:
            score = compute_BM25_score(qt, docno)
            if docno in bm25_scores:
                bm25_scores[docno] += score
            else:
                bm25_scores[docno] = score
    sorted_scores = sorted(bm25_scores.items(), key=lambda kv: kv[1])
    sorted_bm25_scores = collections.OrderedDict(sorted_scores)

    # return the top 50 relevant documents
    ranked_bm25_documents = {}
    i = 0
    for docno, score in reversed(sorted_bm25_scores.items()):
        ranked_bm25_documents[i + 1] = [docno, score]
        i = i + 1
        if i == n:
            break
    return ranked_bm25_documents

# Compute mean precision scores
precision_sum = 0
for query in query_to_pattern_map:
    baseline_retrieved_docs = []
    baseline_rel_docs = get_top_n_relevant_doc(query, 1000)
    for rank, doc in baseline_rel_docs.items():
        baseline_retrieved_docs.append(doc[0])
    relevant_count = 0
    retrieved_bm25_docs = get_top_n_reranked_relevant_docs(query, baseline_retrieved_docs, 50)
    for rank, docno in retrieved_bm25_docs.items():
        if isRelevant(docno[0], query):
            relevant_count = relevant_count + 1
    precision = relevant_count / 50.0
    precision_sum = precision_sum + precision
bm25_mean_precision = precision_sum/100.0
print("BM25 mean precision: ", bm25_mean_precision)   


# Task 3
# Sentence ranking
# construct sentence corpus

#doc_to_sentence_map = {}
#sentence_corpus = {}
#sent_no = 0
#for docno, text in corpus.items():
#    sentences = sent_tokenize(text)
#    for sentence in sentences:
#        sentence_corpus[str(i)] = sentence
#        if docno in doc_to_sentence_map:
#            doc_to_sentence_map[docno].append(str(i))
#        else:
#            l = []
#            l.append(str(i))
#            doc_to_sentence_map[docno] = l
#        i = i + 1
#
#processed_sent_corpus = {}
#for sent_no, sentence in sentence_corpus.items():
#    sentence = str.lower(sentence)
#    table = str.maketrans('', '', string.punctuation)
#    sentence = sentence.translate(table)
#    processed_sent_corpus[sent_no] = nltk.word_tokenize(sentence)
    
# compute count of terms in each sentence
#sentence_count = {}
#for word in vocabulary:
#    sentence_count[word] = 0
#
#for word in vocabulary:
#    for key, value in processed_sent_corpus.items():
#        if word in value:
#            sentence_count[word] = sentence_count[word] + 1
#
# compute term frequencies
#sentence_term_frequencies = {}
#for sentno, sentence in processed_sent_corpus.items():
#    tf = {}
#    freq_count = {}
#    words = []
#    for word in sentence:
#        if word not in words:
#            words.append(word)
#            freq_count[word] = 1
#        else:
#            freq_count[word] += 1
#    for word in words:
#        tf[word] = freq_count[word]
#    sentence_term_frequencies[sentno] = tf

# construct sentence length table
#sentence_length = {}
#for sentno, sentence in processed_sent_corpus.items():
#    sentence_length[sentno] = len(sentence)

# compute average sentence length
#total_sent_length = 0
#for sentno, length in document_length.items():
#    total_sent_length += length
#average_sentence_length = total_sent_length/float(len(processed_sent_corpus))

# construct sentence count
#sentence_count = {}
#sentence_vocabulary = []
#for sentno, sentence in processed_sent_corpus.items():
#    for token in sentence:
#        if token not in sentence_vocabulary:
#            sentence_vocabulary.append(token)
#
#for word in sentence_vocabulary:
#    for sentno, sentence in processed_sent_corpus.items():
#        if word in sentence:
#            if word not in sentence_count:
#                sentence_count[word] = 1
#            else:
#                sentence_count[word] += 1
#
#def get_sentences_from_doc(docs):
#    retrieved_sentences = []
#    for rank, doc in docs.items():
#        sentences = doc_to_sentence_map[doc[0]]
#        retrieved_sentences.append(sentences)
#    return retrieved_sentences
#
# compute bm25 score for sentences
#def compute_sent_BM25_score(qt, sentno):
#    score = 0
#    try:
#        inverse_doc_freq = math.log(((N - sentence_count[qt]+ 0.5)/(sentence_count[qt] + 0.5)) + 1)
#        f_qi_d = sentence_term_frequencies[sentno][qt]
#        score = inverse_doc_freq * f_qi_d * (((k1+1)/(f_qi_d + (k1 * (1-b+(b * sentence_length[sentno]/average_sentence_length))))) + 1)
#    except KeyError:
#        pass
#    return score
#
# get top 50 relevant sentences
#def get_top_n_relevant_sentences(query, bm25_sentences, n):
#    query = str.lower(query)
#    table = str.maketrans('', '', string.punctuation)
#    query = query.translate(table)
#    query_tokens = nltk.word_tokenize(query)
#    sentence_scores = {}
#    for qt in query_tokens:
#        for sentno in bm25_sentences:
#            score = compute_sent_BM25_score(qt, sentno)
#            if sentno in sentence_scores:
#                bm25_scores[sentno] += score
#            else:
#                bm25_scores[sentno] = score
#    sorted_scores = sorted(sentence_scores.items(), key=lambda kv: kv[1])
#    sorted_bm25_scores = collections.OrderedDict(sorted_scores)
#
#    # return the top 50 relevant sentences
#    ranked_sentences = {}
#    i = 0
#    for sentno, score in reversed(sorted_bm25_scores.items()):
#        ranked_sentences[i + 1] = [sentno, score]
#        i = i + 1
#        if i == n:
#            break
#    return ranked_sentences
#
# check if sentence is relevant
#def isSentenceRelevant(sentno, query):
#    text = processed_sent_corpus[sentno]
#    text = sentence_corpus(sentno)
#    query_patterns = query_to_pattern_map[query]
#    for pattern in query_patterns:
#        for token in text:
#            if re.match(pattern, text, flags = re.IGNORECASE):
#                return True
#    return False
#
# compute MRR and mean precision scores
#precision_sum = 0
#query_relevent_ranks = {}
#for query in query_to_pattern_map:
#    baseline_retrieved_docs = []
#    baseline_rel_docs = get_top_n_relevant_doc(query, 1000)
#    for rank, doc in baseline_rel_docs.items():
#        baseline_retrieved_docs.append(doc[0])
#    relevant_count = 0
#    retrieved_bm25_docs = get_top_n_reranked_relevant_docs(query, baseline_retrieved_docs, 50)
#    retrieved_sentences = get_sentences_from_doc(retrieved_bm25_docs)
#    ranked_sentences = get_top_n_relevant_sentences(query, retrieved_sentences, 50)
#    for rank, sent in ranked_sentences.items():
#        if isSentenceRelevant(sent[0], query):
#            print("relevant: ", docno[0], query)
#            relevant_count = relevant_count + 1
#            if query not in query_relevent_ranks:
#                query_relevent_ranks[query] = rank
#            elif query_relevent_ranks[query] > rank:
#                query_relevent_ranks[query] = rank                
#    precision = relevant_count / 50.0
#    precision_sum = precision_sum + precision
#sentence_mean_precision = precision_sum/100.0
#rank_sum = 0
#for query, rank in query_relevent_ranks.items():
#    rank_sum = rank_sum + (1.0/float(rank))
#MRR = rank_sum / 100.0
#print("Sentence mean precision: ", sentence_mean_precision) 
#print("MRR: ", MRR)
#















        
        
        
