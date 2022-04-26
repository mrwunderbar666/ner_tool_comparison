from string import punctuation
import pandas as pd
from nltk.tokenize.treebank import TreebankWordDetokenizer

df = pd.read_feather('datasets/conll/esp.testa.feather')

twd = TreebankWordDetokenizer()

sentences = df.groupby('sentence').token.apply(lambda x: twd.detokenize(x))