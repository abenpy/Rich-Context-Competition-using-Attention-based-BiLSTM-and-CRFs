
import tensorflow as tf
import keras
import os
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt

from keras.models import Model, Input, load_model
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Conv1D, Lambda, GRU
from keras.layers import Bidirectional, concatenate, SpatialDropout1D, GlobalMaxPooling1D
from keras_contrib.layers import CRF

from keras.preprocessing.sequence import pad_sequences
from sklearn_crfsuite.metrics import flat_classification_report
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

import tensorflow as tf
import tensorflow_hub as hub
from keras import backend as K

from keras.callbacks import ModelCheckpoint,TensorBoard,EarlyStopping
from keras import optimizers

import numpy as np
from scipy.spatial.distance import pdist, squareform
from collections import Counter
from gensim.models import FastText
import pickle

from sklearn.utils import shuffle
import re
import random
from itertools import product
import pickle as pkl
from funcy import flatten
from nltk import everygrams, ngrams
import json
from pprint import pprint

from pyemd import emd
from gensim.similarities import WmdSimilarity

# Hyperparams if GPU is available
if tf.test.is_gpu_available():
    BATCH_SIZE = 32  # Number of examples used in each iteration
    EPOCHS = 30  # Number of passes through entire dataset
    MAX_LEN = 30  # Max length of sentence (in words)
    EMBEDDING = 40  # Dimension of word/character embedding vector
    max_len_char = 10

    
# Hyperparams for CPU training
else:
    BATCH_SIZE = 16
    EPOCHS = 5
    MAX_LEN = 30
    EMBEDDING = 20
    max_len_char = 10 

print("BATCH_SIZE is {} ".format(BATCH_SIZE))

class SentenceGetter(object):
    """Class to Get the sentence in this format:
    [(Token_1, Part_of_Speech_1, Tag_1), ..., (Token_n, Part_of_Speech_1, Upper or lower, Tag_1)]"""
    def __init__(self, data):
        """Args:
            data is the pandas.DataFrame which contains the above dataset"""
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]

        self.grouped = self.data.groupby("Sentence_ID_Norm").apply(agg_func)
        self.sentences = [s for s in self.grouped]
    

def Padwords(X,max_len = MAX_LEN):
    new_X = []
    for seq in X:
        new_seq = []
        for i in range(max_len):
            try:
                new_seq.append(seq[i])
            except:
                new_seq.append(word2idx["PAD"])
        new_X.append(new_seq)

    return new_X


def CharGetter(sentences, mlength = max_len_char):
    X_char = []
    for sentence in sentences:
        sent_seq = []
        for i in range(MAX_LEN):
            word_seq = []
            for j in range(mlength):
                try:
                    word_seq.append(char2idx.get(sentence[i][0][j]))
                except:
                    word_seq.append(char2idx.get("PAD"))
            sent_seq.append(word_seq)
        X_char.append(np.array(sent_seq))

    return X_char

def Padtags(y,max_len = MAX_LEN):
    new_y = []
    for seq in y:
        new_seq = []
        for i in range(max_len):
            try:
                new_seq.append(seq[i])
            except:
                new_seq.append(tag2idx["PAD"])
        new_y.append(new_seq)

    return new_y

def softmax(X, theta = 1.0, axis = None):

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(X.shape) if j[1] > 1)

    # multiply y against the theta parameter, 
    X = X * float(theta)
    X = np.exp(X)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(X, axis = axis), axis)
    # finally: divide elementwise
    p = X / ax_sum
    if len(X.shape) == 1: p = p.flatten()

    return p

def euclideanScore(M):
    #Euclidean distance 
    euc_M = squareform(pdist(M, 'euclidean'))
    alpha_euc = softmax(euc_M,  axis = 0)
    return np.sum(alpha_euc,axis=1)

def manhattenScore(M):
    #Manhattan Distance 
    man_M = squareform(pdist(M, 'cityblock'))
    alpha_man = softmax(man_M,  axis = 0)

    return np.sum(alpha_man,axis=1) 

def CosineScore(M):
    cos_M = squareform(pdist(M, 'cosine'))
    alpha_cos = softmax(cos_M,  axis = 0)

    return np.sum(alpha_cos,axis=1) 

# def get_class_weights(y):
#     counter = Counter(y)
#     majority = max(counter.values())
#     return  {cls: float(majority/count) for cls, count in counter.items()}

def build_vocab(all_tokens, tags):
    # Returns:
    # id2token: list of tokens, where id2token[i] returns token that corresponds to token i
    # token2id: dictionary where keys represent tokens and corresponding values represent indices
    token_counter = Counter(all_tokens)
    vocab, count = zip(*token_counter.most_common(max_vocab_size))
    vocab= list(set((list(vocab) + tags)))
    if 'UNK' in vocab:
        vocab.remove('UNK')
    if 'PAD' in vocab:
        vocab.remove('PAD')
    id2token = vocab
    token2id = dict(zip(vocab, range(2,2+len(vocab)))) #good way to creat new dic 
    id2token = ["PAD", "UNK"] + id2token
    token2id["PAD"] = 0 
    token2id["UNK"] = 1
    return token2id, id2token


def subdata_getter(pub_ids,data,batch_size = BATCH_SIZE, alpha_t = True):
    subdata = {}
    subdata['words'],subdata['chars'],subdata['tags'], subdata['alpha'] = {},{},{},{}
    for pub_id in pub_ids:
        eachpub_sentence = {}
        #sentences
        df = data.loc[data["Pub_id"]==pub_id]

        each_sentences = SentenceGetter(df).sentences 
        padsent_num = int(np.ceil(len(each_sentences) /(batch_size))*batch_size - len(each_sentences))
        for i in range(padsent_num):
            each_sentences.append([('PAD', 'PAD', 'PAD')])
        #words
        each_words = [[word2idx[w[0]] for w in s] for s in each_sentences]
  
        subdata['words'][pub_id] = Padwords(each_words)
        #characters
        subdata['chars'][pub_id] = CharGetter(each_sentences)

        #tags
        y = [[tag2idx[w[2]] for w in s] for s in each_sentences]  
        y = Padtags(y)
        #one hot encoder
        subdata['tags'][pub_id] = [to_categorical(i, num_classes=n_tags+1) for i in y] 
            
        if alpha_t:
            #get alpha
            word_embedding_pub = [[embedding_matrix[j] for j in s] for s in subdata['words'][pub_id]]
            word_embedding_pub = np.array(word_embedding_pub).reshape(-1,100)

            #use ConsineScore
            alpha = CosineScore(word_embedding_pub)
    #         alpha = alpha_func.CosineScore
            # alpha = alpha.reshape(-1,MAX_LEN,1024)
            subdata['alpha'][pub_id] = alpha.tolist()

    return subdata


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, pub_IDs, data_forgene, batch_size=BATCH_SIZE,
                 shuffle_pub=True, shuffle_sent=True, Type = 'train', alpha = True):
        'Initialization'
        
        self.batch_size = batch_size
        self.pub_IDs = pub_IDs
        self.data_forgene = data_forgene
        self.shuffle_pub = shuffle_pub
        self.shuffle_sent = shuffle_sent
        self.type = Type
        self.alpha = alpha
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        length = 0
        for i in self.pub_IDs:
            length = length + int(len(self.data_forgene['words'][i]) / self.batch_size)
        return length

    def __getitem__(self, index):
        'Generate one batch of data'
        # generator from self.shuffled_data from second epoch
    
#         if self.shuffled_data:
        X = self.shuffled_data['words'][index*self.batch_size:(index+1)*self.batch_size]
        char = self.shuffled_data['chars'][index*self.batch_size:(index+1)*self.batch_size]
        y = self.shuffled_data['tags'][index*self.batch_size:(index+1)*self.batch_size]
        y = np.array(y).reshape(len(y), MAX_LEN, 5)
        
        if self.alpha:
            alpha = self.shuffled_data['alpha'][index*self.batch_size:(index+1)*self.batch_size]
            input_data = [np.array(X), np.array(char), np.array(alpha)]
            
        else:
            input_data = [np.array(X), np.array(char)]

        if self.type == 'train':    
            return input_data, y
        else: 
            return input_data

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.pub_IDs))
        if self.shuffle_pub == True:
            np.random.shuffle(self.indexes)     
            pub_IDs_temp = [self.pub_IDs[k] for k in self.indexes]
            #shuffle pubs
            shuffled_data = {}
            shuffled_data['words'],shuffled_data['chars'],shuffled_data['tags'], shuffled_data['alpha']= [], [], [],[]
            #sentences of each pubs can be moded by batch_size
            for i in pub_IDs_temp:     
                batch_length = len(self.data_forgene['tags'][i])
                if self.shuffle_sent == True:
                    indexes = np.arange(batch_length)
                    np.random.shuffle(indexes)
                    shuffled_data['words'] = shuffled_data['words'] + np.take(self.data_forgene['words'][i], indexes, axis=0).tolist()
                    shuffled_data['chars']= shuffled_data['chars'] + np.take(self.data_forgene['chars'][i], indexes, axis=0).tolist()
                    shuffled_data['tags'] = shuffled_data['tags'] + np.take(self.data_forgene['tags'][i], indexes, axis=0).tolist()
                    if self.alpha:
                        shuffled_data['alpha'] = shuffled_data['alpha'] + np.take(self.data_forgene['alpha'][i], indexes, axis=0).tolist()
                else:         
                    shuffled_data['words'] = shuffled_data['words']+self.data_forgene['words'][i]
                    shuffled_data['chars']= shuffled_data['chars']+self.data_forgene['chars'][i]
                    shuffled_data['tags'] = shuffled_data['tags']+self.data_forgene['tags'][i]
                    if self.alpha:
                        shuffled_data['alpha'] = shuffled_data['alpha']+self.data_forgene['alpha'][i]

        else:
            shuffled_data = self.data_forgene
            
        self.shuffled_data = shuffled_data
        
        
class NBatchLogger(keras.callbacks.Callback):
    """
    A Logger that log average performance per `display` steps and saved loss & acc by batches 
    """ 
    def __init__(self, display=500):
        self.step = 0
        self.display = display
        self.val = val_generator
        self.metric_cache = {}
        self.train_log = {}
        self.val_log = {}

    def on_train_begin(self, logs={}):
        self.val_log['loss'],self.val_log['acc'] = [], []
        self.train_log['loss'],self.train_log['acc'] = [], []

    def on_batch_end(self, batch, logs={}):
        self.step += 1
        for k in self.params['metrics']:
            if k in logs:
                self.metric_cache[k] = self.metric_cache.get(k, 0) + logs[k]
        if self.step % self.display == 0:
            metrics_log = ''
            val_loss, val_acc = self.model.evaluate_generator(generator = val_generator)
            i = 0
            for (k, v) in self.metric_cache.items():
                val = v / self.display
                if abs(val) > 1e-3:
                    metrics_log += ' - %s: %.4f' % (k, val)
                else:
                    metrics_log += ' - %s: %.4e' % (k, val)
                if i == 0:
                    self.train_log['loss'].append(val) 
                else:
                    self.train_log['acc'].append(val) 
                i+=1

            self.val_log['loss'].append(val_loss)
            self.val_log['acc'].append(val_acc)
            
            print('step: {}/{} ... {}'.format(self.step,
                                          self.params['steps'],
                                          metrics_log))
            
            self.metric_cache.clear()

#Load Lookup table used in the training
####################################################################################################################################
data =  pd.read_csv('df_concat.csv', encoding="latin1").fillna(method="ffill")
pub_ids = list(set(data["Pub_id"].values))

data_tags = data.loc[(data['Tag'] == 'B'),"Word"].values.tolist() + \
           data.loc[(data['Tag'] == 'I'),"Word"].values.tolist() + \
              data.loc[(data['Tag'] == 'M'),"Word"].values.tolist()

all_words = data["Word"].values.tolist()
max_vocab_size = 50000

#words
n_words = max_vocab_size + 2
# Vocabulary Key:word -> Value:token_index
# The first 2 entries are reserved for PAD and UNK
word2idx, idx2word = build_vocab(all_words, data_tags)
words = idx2word
print("Number of words in the dataset: ", n_words)

#change all other words in dataframe as "UNK"
data.loc[~data['Word'].isin(words), 'Word'] = 'UNK'

#tags
tags = list(set(data["Tag"].values))
print("Tags:", tags)
n_tags = len(tags)
print("Number of Labels: ", n_tags)

#characters
chars = set([w_i for w in words for w_i in w]) 
n_chars = len(chars)
print("Number of Labels: ", n_chars)
  
tag2idx = {t: i+1 for i, t in enumerate(tags)}
tag2idx["PAD"] = 0
# Vocabulary Key:tag_index -> Value:Label/Tag
idx2tag = {i: w for w, i in tag2idx.items()}

# Char Key:char -> Value:token_index
char2idx = {c: i + 2 for i, c in enumerate(chars)}
char2idx["UNK"] = 1
char2idx["PAD"] = 0

#Load model and data
####################################################################################################################################
model = load_model("ner_ch_o.h5")

#Load test data
data_te =  pd.read_csv('df_concat_test.csv', encoding="latin1").fillna(method="ffill")
data_te.loc[~data_te['Word'].isin(words),'Word'] = "UNK"
te_pub = list(set(data_te["Pub_id"].values))
test = subdata_getter(te_pub,data_te)
te_generator =  DataGenerator(te_pub, test, Type = 'test')


#Eval of NER
####################################################################################################################################
#get true tags
y_temp = test['tags']
y_te = []
# y_index = {}
# word_temp = test['words']
# word_te = []
for pub_id in te_pub: 
    y_te = y_te + y_temp[pub_id]

pred_cat = model.predict_generator(generator=te_generator, verbose=0)
pred = np.argmax(pred_cat, axis=-1)
y_te_true = np.argmax(y_te, -1)

# Convert the index to tag
pred_tag = [[idx2tag[i] for i in row] for row in pred]
y_te_true_tag = [[idx2tag[i] for i in row] for row in y_te_true] 

report = flat_classification_report(y_pred=pred_tag, y_true=y_te_true_tag)
pickle.dump(prediction, open('report_ner', 'wb'))
print(report)

#Eval of Dataset ID
####################################################################################################################################
import string
#stopwords list
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
nltk.download("stopwords")
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

words_fast =FastText.load('model_fast30/model_fast.model') 

def tokenize(sent):
#     if (token.text not in punctuations)
    tokens_lower = [token.lower() for token in sent if (token not in punctuations) 
                    and (token not in stop_words)]
    tokens_lemma = [wnl.lemmatize(word) for word in tokens_lower]
    return tokens_lemma

prediction = {}
Y = {}
Words = {}

#load whole dataset ID and mentionlist
mentionlist = {}
with open('./train_test/data_sets.json') as f:
    dataset = json.load(f)
for i in range(len(dataset)):
    mentionlist[dataset[i]["data_set_id"]]=dataset[i]["mention_list"]

mentionlist100 = {}
with open('./train_test/data_set_citations_100.json') as f:
    dataset100 = json.load(f)
for i in range(len(dataset100)):
    mentionlist100[dataset100[i]["publication_id"]]=dataset100[i]["data_set_id"]

ID_true = []
predicted_ID = []

for pub_id in te_pub:
    each_test = subdata_getter([pub_id],data_te)
    each_te_generator =  DataGenerator([pub_id], each_test, Type = 'test')
    pred_cat = model.predict_generator(generator=each_te_generator, verbose=0)
    pred = np.argmax(pred_cat, axis=-1)
    
    #get words
    word_temp = each_test['words']
    word_flatten = np.array(word_temp[pub_id]).flatten()
    y_flatten = pred.flatten()
    
    pred_ind_B = list(np.where(y_flatten == tag2idx['B']) [0])
    pred_ind_I = list(np.where(y_flatten == tag2idx['I']) [0])
    pred_ind = pred_ind_B + pred_ind_I
  
#   pred_ind = np.sort(np.array(pred_ind))
    prediction[pub_id] = {}
    if len(pred_ind) == 0:
        prediction[pub_id]['predict_word']='Can not find mention list'
    else:
        prediction[pub_id]['predict_word'] = [idx2word[i] for i in np.take(word_flatten,pred_ind)]


    Y[pub_id] = pred
    Words[pub_id] = word_temp[pub_id]
    #compute similarity between predicted mentionlist with the whole data set and find the maximum, assign
    #this dataset id as prediction
    max_score = 0
    for i in mentionlist:
        reference = mentionlist[i]
        if len(reference)>0:
            #remove stop words
            cleaned_pred = tokenize(prediction[pub_id]['predict_word'])
            cleaned_ref = tokenize(reference)
            for i in cleaned_ref:
                similarity = words_fast.wv.wmdistance(cleaned_pred, i)
                if similarity>max_score:
                    max_score = similarity
                    prediction[pub_id]['predict_id'] = i
                    predicted_ID.append(i)
    ID_true.append(mentionlist100[pub_id])


report_ID = flat_classification_report(y_pred=predicted_ID, y_true=ID_true)
print(report_ID)

pickle.dump(prediction, open('prediction', 'wb'))
pickle.dump(Y, open('Y_prediction', 'wb'))
pickle.dump(Words, open('Words_prediction', 'wb'))

#     y_te = y_te + y_temp[pub_id]
#     y_index[pub_id] = np.argmax(y_temp[pub_id], -1)
    #get true words
