import tensorflow as tf
import keras
import os
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt

from keras.models import Model, Input, load_model
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Conv1D, Lambda
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

# Hyperparams if GPU is available
if tf.test.is_gpu_available():
    BATCH_SIZE = 32  # Number of examples used in each iteration
    EPOCHS = 30  # Number of passes through entire dataset
    MAX_LEN = 30  # Max length of sentence (in words)
    EMBEDDING = 40  # Dimension of word/character embedding vector
    max_len_char = 10

    
# Hyperparams for CPU training
else:
    BATCH_SIZE = 8
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

#     # subtract the max for numerical stability
#     y = y - np.expand_dims(np.max(y, axis = axis), axis)
    
    # exponentiate y
    X = np.exp(X)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(X, axis = axis), axis)

    # finally: divide elementwise
    p = X / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p

def euclideanScore(M):
    #Euclidean distance 
    euc_M = squareform(pdist(M, 'euclidean'))
    alpha_euc = softmax(euc_M,  axis = 0)
    return np.sum(alpha_euc,axis=1)

def manhattenScore(M):
    man_M = squareform(pdist(M, 'cityblock'))
    alpha_man = softmax(man_M,  axis = 0)

    return np.sum(alpha_man,axis=1) 

    #Manhattan Distance 

def CosineScore(M):

    cos_M = squareform(pdist(M, 'cosine'))
    alpha_cos = softmax(cos_M,  axis = 0)

    return np.sum(alpha_cos,axis=1) 


def get_class_weights(y):
    counter = Counter(y)
    majority = max(counter.values())
    return  {cls: float(majority/count) for cls, count in counter.items()}


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


def subdata_getter(pub_ids,data,batch_size = BATCH_SIZE, alpha = False):
    subdata = {}
    subdata['words'],subdata['chars'],subdata['tags'], subdata['alpha'] = {},{},{},{}
    for pub_id in pub_ids:
        eachpub_sentence = {}
        #sentences
        df = data.loc[data["Pub_id"]==pub_id]

        #pad df rows till can be divide by batch_size
        padlength = int(np.ceil(df.shape[0] /batch_size)*batch_size - df.shape[0])
        add_Word, add_POS, add_Tag = [], [], []
        for i in range(padlength):
            add_Word.append("PAD")
            add_POS.append("PAD")
            add_Tag.append("PAD")
            
        add_rows = {'Word': add_Word,
                   'POS': add_POS,
                   'Tag': add_Tag}
        df_add = pd.DataFrame.from_dict(add_rows)
        df_padded = pd.concat([df,df_add]).fillna(method="ffill")

        each_sentences = SentenceGetter(df_padded).sentences 
        
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
        
        if alpha:
            #get alpha
            word_embedding_pub = [[embedding_matrix[j] for j in s] for s in subdata['words'][pub_id]]
            word_embedding_pub = np.array(word_embedding_pub).reshape(-1,300)

            #use ConsineScore
            alpha = CosineScore(word_embedding_pub)
    #         alpha = alpha_func.CosineScore
            # alpha = alpha.reshape(-1,MAX_LEN,1024)
            subdata['alpha'][pub_id] = alpha.tolist()

    return subdata


class NBatchLogger(keras.callbacks.Callback):
    """
    A Logger that log average performance per `display` steps.
    """
    
    def __init__(self, display=10):
        self.step = 0
        self.display = display
        self.val = val_generator
        self.metric_cache = {}

    def on_train_begin(self, logs={}):
        self.train_log = {}
        self.val_log = {}
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

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, pub_IDs, data_forgene, batch_size=BATCH_SIZE,
                 shuffle_pub=True, shuffle_sent=True, Type = 'train', alpha = False):
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
                    print(len(self.data_forgene['alpha'][i]))
        else:
            shuffled_data = self.data_forgene
            
        self.shuffled_data = shuffled_data


class NBatchLogger(keras.callbacks.Callback):
    """
    A Logger that log average performance per `display` steps.
    """
    
    def __init__(self, display=500):
        self.step = 0
        self.display = display
        self.val = val_generator
        self.metric_cache = {}

    def on_train_begin(self, logs={}):
        self.train_log = {}
        self.val_log = {}
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

data =  pd.read_csv('df_concat.csv', encoding="latin1").fillna(method="ffill")
data = data[:3867199]
pub_ids = list(set(data["Pub_id"].values))
data_tags = data.loc[(data['Tag'] == 'B'),"Word"].values.tolist() + \
           data.loc[(data['Tag'] == 'I'),"Word"].values.tolist() + \
              data.loc[(data['Tag'] == 'M'),"Word"].values.tolist()

all_words = data["Word"].values.tolist()
max_vocab_size = 30000

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

words_fast =FastText.load('model_fast30/model_fast.model') 

#load pretrained word embedding
embedding_matrix = np.ones((len(word2idx), 100),dtype='float32')
embedding_matrix[0] = np.zeros(100,dtype='float32')
# with open('wiki-news-300d-1M.vec') as f:
for i in range(2,len(idx2word)-2):
    embedding_matrix[i] = words_fast[idx2word[i]]
#         ordered_words_ft.append(s[0])
print('Found %s word vectors.' % len(embedding_matrix))

# for word, i in word2idx.items():
#     embedding_vector = embeddings_index.get(word)
#     if embedding_vector is not None:
#         # words not found in embedding index will be all-zeros.
#         embedding_matrix[i] = embedding_vector



# Model definition
# input and embedding for words

word_in = Input(shape=(MAX_LEN,))

word_embedding = Embedding( input_dim=len(word2idx), output_dim = 100,
                            weights=[embedding_matrix],
                            input_length=MAX_LEN,
                            trainable=True)(word_in)


# input and embeddings for characters
char_in = Input(shape=(MAX_LEN, max_len_char,))
emb_char = TimeDistributed(Embedding(input_dim=n_chars + 2, output_dim=10,
                           input_length=max_len_char, mask_zero=True))(char_in)

# character LSTM to get word encodings by characters
char_enc = TimeDistributed(LSTM(units=20, return_sequences=False,
                                recurrent_dropout=0.5))(emb_char)

# main LSTM
x = concatenate([word_embedding, char_enc])
x = SpatialDropout1D(0.3)(x)

main_lstm = Bidirectional(LSTM(units=50, return_sequences=True,
                               recurrent_dropout=0.6))(x)  #dropout 0.1试试？
model = TimeDistributed(Dense(50, activation="relu"))(main_lstm)

crf = CRF(n_tags+1)  # CRF layer, n_tags+1(PD)
out = crf(model)  # output

# out = Lambda(lambda x: K.reshape(x,(-1,5)))(out)
model = Model([word_in, char_in], out)

# set optimizer 
# rmsprop = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=1e-5)

adam = optimizers.Adam(lr=0.0001, epsilon=None, decay=1e-6)
model.compile(optimizer=adam, loss=crf.loss_function, metrics=[crf.accuracy]) #use crf
model.summary()
#sample_weight_mode="temporal"


tr_pubs = pub_ids[:int(len(pub_ids)*0.9)]
val_pubs = pub_ids[int(len(pub_ids)*0.9):] 

train = subdata_getter(tr_pubs,data)
validation = subdata_getter(val_pubs,data)


tr_generator =  DataGenerator(tr_pubs,train)
val_generator =  DataGenerator(val_pubs,validation)

history = NBatchLogger()
model.fit_generator(generator=tr_generator,shuffle=False, epochs=10, verbose=0,callbacks=[history]) #,callbacks=callbacks_list
model.evaluate_generator(generator = val_generator, use_multiprocessing=True, verbose=0)

history_save= []
history_save.append(history.train_log)
history_save.append(history.val_log)
pickle.dump(history_save, open('history_adam_0.0001', 'wb'))

#unknow words to "UNK"
data_te =  pd.read_csv('df_concat_test.csv', encoding="latin1").fillna(method="ffill")
data_te.loc[~data_te['Word'].isin(words),'Word'] = "UNK"
# data_te = data_te[:20000]

te_pub = list(set(data_te["Pub_id"].values))


test = subdata_getter(te_pub,data_te)
te_generator =  DataGenerator(te_pub, test, Type = 'test')

#get true tags
y_temp = test['tags']
y_te = []
for pub_id in te_pub:
    length=int(np.floor(len(y_temp[pub_id]) /BATCH_SIZE )*BATCH_SIZE)
    y_te=y_te+ y_temp[pub_id][:length]

# Eval
pred_cat = model.predict_generator(generator=te_generator, verbose=0)
pred = np.argmax(pred_cat, axis=-1)
y_te_true = np.argmax(y_te, -1)

# Convert the index to tag
pred_tag = [[idx2tag[i] for i in row] for row in pred]
y_te_true_tag = [[idx2tag[i] for i in row] for row in y_te_true] 

report = flat_classification_report(y_pred=pred_tag, y_true=y_te_true_tag)
print(report)


