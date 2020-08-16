import xml.etree.ElementTree as ElementTree
import re
import collections
import nltk
import string
import math
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


print(list(idf.items())[:100])
        
