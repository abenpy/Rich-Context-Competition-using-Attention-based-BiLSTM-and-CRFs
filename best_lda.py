import os
import sys
import nltk
import gensim
import re
import csv
import json
import spacy
import numpy as np
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from gensim.corpora import Dictionary
#from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial
nltk.download('words')
#spacy.load('en_core_web_md')
nlp = spacy.load('en')#spacy.load('en_core_web_md')
nltk.download('wordnet')
np.random.seed(2018)
wordlist = [w for w in nltk.corpus.words.words('en') if w.islower()]
data_path = sys.argv[1]
filenames=[]
ids=[]
with open(data_path+'/publications.json') as json_data:
    data = json.load(json_data)
    for i in range(len(data)):
        filenames.append(data[i]["text_file_name"])
        ids.append(data[i]["publication_id"])
#path=sys.argv[0]
# path="/scratch/ruggles/nlp_competition/text/"
###functions to lemmatize and preprocess the text
def lemmatize(text):
    return WordNetLemmatizer().lemmatize(text, pos='v')
def preprocess(text):
    result=[]
    for token in gensim.utils.simple_preprocess(text) :
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            token=lemmatize(token)
            if token in wordlist:
                result.append(token)
    return result
###preprocess the text and remove tokens with frequence less than 3
prepro_docs=[]
for file in filenames:
    # print(file)
    # print(data_path+'/files/text/'+file)
    with open(data_path+'/files/text/'+file, 'r', encoding="utf-8") as f:
        my_file=f.read()
    my_file=my_file[:10000]
    prepro=preprocess(my_file)
    prepro_docs.append(prepro)
frequency = defaultdict(int)
for text in prepro_docs:
    for token in text:
        frequency[token] += 1#
text = [[token for token in text if frequency[token] > 3]
        for text in prepro_docs]
#get the embeddings of the preprocessed text
dictionary=Dictionary(text)
bow_corpus = [dictionary.doc2bow(doc) for doc in text]
#train the model
lda_model =  gensim.models.LdaMulticore(bow_corpus,
                                   num_topics =30, ###change this to 30
                                   id2word = dictionary,                                    
                                   passes = 10,
                                   workers = 2,
                                   minimum_probability=0)
#compute the coherence score of the model
coherence_model_lda = gensim.models.CoherenceModel(model=lda_model, texts=text, dictionary=dictionary, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)
####get embeddings for the main topics of sage database and add other topics to it
custom_unique=['geography environmental science', 'business management economy', 'social study sociology', 
                 'health care hospitalization', 'criminology criminal justice', 'bank finance economy',
                 'media communication culture', 'health social care', 'psychological study psychology',
                 "counseling psychotherapy advisement", "politics political sciences",
                 "linguistics languages translation"]
processed_custom=[]
for i in range(len(custom_unique)):
    processed_custom.append(preprocess(str(custom_unique[i])))
custom_vector=[]
for i in range(len(processed_custom)):
    field1=" ".join(processed_custom[i])
    custom_vector.append(nlp(field1).vector)
primary_topics=processed_custom
primary_topics_vector=custom_vector
##get the embeddings for the words in the identified topics, 
## as well as the top topic score and idenx for each document 
my_score=[]
my_index=[]
top_topic_words=[]
top_topic_vector=[]
for i in range(len(filenames)):
    index=sorted(lda_model[bow_corpus[i]], key=lambda tup: -1*tup[1])[0][0]
    score=sorted(lda_model[bow_corpus[i]], key=lambda tup: -1*tup[1])[0][1]
    my_index.append(index)
    my_score.append(score)

my_topics=lda_model.print_topics()
for i in range(len(my_topics)):
    topic=str(my_topics[i])
    top_words=re.findall('"([^"]*)"', topic)
    words=top_words[:4]
    #print(words)
    top_topic_words.append(" ".join(words))
#print(top_topic_words)
for i in range(len(top_topic_words)):
    top_topic_vector.append(nlp(top_topic_words[i]).vector)
###Calculate the cosine similarity and get the real topic name 
assigned_topics=[]
for i in range(len(top_topic_vector)):
    each_doc_similarities=[]
    for j in range(len(primary_topics_vector)):
        each_doc_similarities.append(1 - spatial.distance.cosine(top_topic_vector[i],primary_topics_vector[j]))
    top_topic_index=each_doc_similarities.index(max(each_doc_similarities))
    assigned_topics.append(" ".join(primary_topics[top_topic_index]))
###get the real topic for each document
for i in range(len(assigned_topics)):
    for j in range(len(my_index)):
        if my_index[j]==i:
            my_index[j]=assigned_topics[i]
###Save the json output file
my_list=[]
for i in range(len(my_score)):
    my_dict = {
        "publication_id": ids[i], 
        "score": str(my_score[i]), 
        "research_field": my_index[i]
    }
    my_list.append(my_dict)
#print(my_list)
with open(data_path+'../output/research_fields.json', 'w') as fout:
    json.dump(my_list, fout)

