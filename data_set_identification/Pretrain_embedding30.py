from gensim.models import FastText
import pandas as pd

data =  pd.read_csv('df_concat.csv', encoding="latin1").fillna(method="ffill")

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

sentences = SentenceGetter(data).sentences 
#words
words = [[ w[0] for w in s] for s in sentences]

model_fast = FastText(words, size=100, window=5, min_count=1, workers=4,sg=1,iter=30) #skip-gram to train

model_fast.save("model_fast30/model_fast.model")

# test the training result
existent_word = "MiDi"
print(existent_word in model_fast.wv.vocab)
print(model_fast['MiDi'])