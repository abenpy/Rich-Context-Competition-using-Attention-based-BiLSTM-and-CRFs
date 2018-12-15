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
from keras_contrib.utils import save_load_utils

import tensorflow as tf
import tensorflow_hub as hub
from keras import backend as K

from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras import optimizers

# Hyperparams if GPU is available
if tf.test.is_gpu_available():
    BATCH_SIZE = 64  # Number of examples used in each iteration
    EPOCHS = 5  # Number of passes through entire dataset
    MAX_LEN = 30  # Max length of sentence (in words)
    EMBEDDING = 40  # Dimension of word/character embedding vector
    max_len_char = 15 

    
# Hyperparams for CPU training
else:
    BATCH_SIZE = 32
    EPOCHS = 5
    MAX_LEN = 30
    EMBEDDING = 20
    max_len_char = 15 

print("BATCH_SIZE is {} ".format(BATCH_SIZE))


data =  pd.read_csv('df_concat.csv', encoding="latin1").fillna(method="ffill")

#words
words = list(set(data["Word"].values))
n_words = len(words)
print("Number of words in the dataset: ", n_words)

#tags
tags = list(set(data["Tag"].values))
print("Tags:", tags)
n_tags = len(tags)
print("Number of Labels: ", n_tags)

#characters
chars = set([w_i for w in words for w_i in w]) 
n_chars = len(chars)
print("Number of Labels: ", n_chars)

# Vocabulary Key:word -> Value:token_index
# The first 2 entries are reserved for PAD and UNK
word2idx = {w: i + 2 for i, w in enumerate(words)}
word2idx["UNK"] = 1 # Unknown words
word2idx["PAD"] = 0 # Padding
# Vocabulary Key:token_index -> Value:word
idx2word = {i: w for w, i in word2idx.items()}

# Vocabulary Key:Label/Tag -> Value:tag_index
# The first entry is reserved for PAD
tag2idx = {t: i+1 for i, t in enumerate(tags)}
tag2idx["PAD"] = 0
# Vocabulary Key:tag_index -> Value:Label/Tag
idx2tag = {i: w for w, i in tag2idx.items()}

# Char Key:char -> Value:token_index
char2idx = {c: i + 2 for i, c in enumerate(chars)}
char2idx["UNK"] = 1
char2idx["PAD"] = 0

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

        self.grouped = self.data.groupby("Sentence_ID").apply(agg_func)
        self.sentences = [s for s in self.grouped]
        
getter = SentenceGetter(data)

# Get all the sentences
sentences = getter.sentences

# Convert each sentence from list of Token to list of word_index
X_word = [[word2idx[w[0]] for w in s] for s in sentences]
# Padding each sentence to have the same lenght
X_word = pad_sequences(maxlen=MAX_LEN, sequences=X_word, value=word2idx["PAD"], padding='post', truncating='post')

# Convert char to index
X_char = []
for sentence in sentences:
    sent_seq = []
    for i in range(MAX_LEN):
        word_seq = []
        for j in range(max_len_char):
            try:
                word_seq.append(char2idx.get(sentence[i][0][j]))
            except:
                word_seq.append(char2idx.get("PAD"))
        sent_seq.append(word_seq)
    X_char.append(np.array(sent_seq))


# Convert Tag/Label to tag_index
y = [[tag2idx[w[2]] for w in s] for s in sentences]
# Padding each sentence to have the same lenght
y = pad_sequences(maxlen=MAX_LEN, sequences=y, value=tag2idx["PAD"], padding='post', truncating='post')

# One-Hot encode
y = [to_categorical(i, num_classes=n_tags+1) for i in y]  # n_tags+1(PAD)  #原blog未涉及，先加上


# X_word_tr, X_word_te, y_tr, y_te = train_test_split(X_word, y, test_size=0.1, random_state=2018)
# X_char_tr, X_char_te, _, _ = train_test_split(X_char, y, test_size=0.1, random_state=2018)

X_word_tr = X_word
X_char_tr = X_char
y_tr = y

# Model definition
# input and embedding for words
word_in = Input(shape=(MAX_LEN,))
emb_word = Embedding(input_dim=n_words + 2, output_dim=20,
                     input_length=MAX_LEN, mask_zero=True)(word_in)

# input and embeddings for characters
char_in = Input(shape=(MAX_LEN, max_len_char,))
emb_char = TimeDistributed(Embedding(input_dim=n_chars + 2, output_dim=10,
                           input_length=max_len_char, mask_zero=True))(char_in)

# character LSTM to get word encodings by characters
char_enc = TimeDistributed(LSTM(units=20, return_sequences=False,
                                recurrent_dropout=0.5))(emb_char)

# main LSTM
x = concatenate([emb_word, char_enc])
x = SpatialDropout1D(0.3)(x)
main_lstm = Bidirectional(LSTM(units=50, return_sequences=True,
                               recurrent_dropout=0.6))(x)  #dropout 0.1试试？
model = TimeDistributed(Dense(50, activation="relu"))(main_lstm)

#新增crf 可能有问题
crf = CRF(n_tags+1)  # CRF layer, n_tags+1(PAD)

out = crf(model)  # output
model = Model([word_in, char_in], out)
rmsprop = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=1e-5)
model.compile(optimizer=rmsprop, loss=crf.loss_function, metrics=[crf.accuracy]) #use crf
model.summary()

# #early stop
# filepath="ner_ch_em_v4.hdf5"
# checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=2, save_best_only=True, mode='min',save_weights_only=True) #val_acc
# early_stop = EarlyStopping(monitor='val_loss', patience=5, mode='min') 
# callbacks_list = [checkpoint, early_stop]

history = model.fit([X_word_tr,
                     np.array(X_char_tr).reshape((len(X_char_tr), MAX_LEN, max_len_char))],
                    np.array(y_tr).reshape(len(y_tr),MAX_LEN, 5),
                    batch_size=BATCH_SIZE, epochs=3, validation_split=0.1, verbose=2,shuffle=True)  #,callbacks=callbacks_list

# #save json
# model_json = model.to_json()
# with open("ner_ch_em_v4_json.json", "w") as json_file:
#     json_file.write(model_json)

# # serialize weights to HDF5
# model.save('ner_ch_em_v4_json.h5')
# print("Saved model to disk")
#save_load_utils.save_all_weights(model,'ner_ch_em_v4.h5')


data_te =  pd.read_csv('df_concat_test.csv', encoding="latin1").fillna(method="ffill")
data_te = data_te.loc[data_te['Word'].isin(words)]

getter = SentenceGetter(data_te)

# # Get all the sentences
each_sentences = getter.sentences

# # Convert each sentence from list of Token to list of word_index
X_word  = [[word2idx[w[0]] for w in s] for s in each_sentences] 
# # Padding each sentence to have the same lenght
X_word_te = pad_sequences(maxlen=MAX_LEN, sequences=X_word, value=word2idx["PAD"], padding='post', truncating='post')


# # Convert char to index
X_char_te = []
for sentence in each_sentences:
    sent_seq = []
    for i in range(MAX_LEN):
        word_seq = []
        for j in range(max_len_char):
            try:
                word_seq.append(char2idx.get(sentence[i][0][j]))
            except:
                word_seq.append(char2idx.get("PAD"))
        sent_seq.append(word_seq)
    X_char_te.append(np.array(sent_seq))
    
# # true y
y = [[tag2idx[w[2]] for w in s] for s in each_sentences]
# # Padding each sentence to have the same lenght
y = pad_sequences(maxlen=MAX_LEN, sequences=y, value=tag2idx["PAD"], padding='post', truncating='post')

# # One-Hot encode
y_te = [to_categorical(i, num_classes=n_tags+1) for i in y]  # n_tags+1(PAD)  

# # Eval
pred_cat = model.predict([X_word_te,
                     np.array(X_char_te).reshape((len(X_char_te), MAX_LEN, max_len_char))])
pred = np.argmax(pred_cat, axis=-1)
y_te_true = np.argmax(y_te, -1)

# # Convert the index to tag
pred_tag = [[idx2tag[i] for i in row] for row in pred]
y_te_true_tag = [[idx2tag[i] for i in row] for row in y_te_true] 

report = flat_classification_report(y_pred=pred_tag, y_true=y_te_true_tag)
print(report)




