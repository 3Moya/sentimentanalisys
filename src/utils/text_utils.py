import re
import demoji
import spacy
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

spcy = spacy.load('en_core_web_sm')

def remove_mentions(text: str):
    return re.sub(r'@(\w+)', '', text)

def remove_urls(text: str):
    return re.sub(
        r'((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*',
        '', text
    )

def remove_symbols(text):
    return re.sub(r'[^\w\s]', '', text)

def remove_emojis(text: str):
    return demoji.replace(text, '')

def lemmatization(text: str):
    return ' '.join([word.lemma_.lower() for word in spcy(text) if not word.is_stop])

def train_tokenizer(values: np.array):
    tokenizer = Tokenizer(num_words=3000, split=' ')
    tokenizer.fit_on_texts(values)

    return tokenizer

def vectorization(tokenizer: Tokenizer, values: np.array):
    return pad_sequences(tokenizer.texts_to_sequences(values))

def zero_pad(values: np.array, length: int):
    return np.lib.pad(values, ((0, 0), (length - values.shape[1], 0)), 'constant', constant_values=(0))