import xml.etree.ElementTree as ElementTree
import re
import collections
import nltk
import string
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

for key, value in processed_corpus. items():
    print(key, ' : ', value)
        
