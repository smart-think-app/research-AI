import spacy
from keras.preprocessing.text import Tokenizer
import numpy as np
from keras.utils import to_categorical


def read_file(filepath):
    with open(filepath) as f:
        str_text = f.read()

    return str_text


nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger", "ner"])
nlp.max_length = 1000000


def separate_punc(doc_text):
    return [token.text.lower() for token in nlp(doc_text) if
            token.text not in '\n\n \n\n\n!"-#$%&()--.*+,-/:;<=>?@9[\\]^_`{|}~\t\n ']


d = read_file("topic.txt")

tokens = separate_punc(d)

train_len = 25 + 1

text_sequence = []
for i in range(train_len, len(tokens)):
    seq = tokens[i - train_len:i]
    text_sequence.append(seq)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_sequence)

sequences = tokenizer.texts_to_sequences(text_sequence)

vocabulary_size = len(tokenizer.word_counts)
# print(tokenizer.index_word)
# for i in sequences[0]:
#     print("{} - {}".format(i,tokenizer.index_word[i]))
sequences = np.array(sequences)

X = sequences[:, :-1]
y = sequences[:, -1]

y = to_categorical(y, num_classes=vocabulary_size + 1)

seq_len = X.shape[1]

from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding


def create_model(vocabulary_size, seq_len):
    model = Sequential()
    model.add(Embedding(vocabulary_size, seq_len, input_length=seq_len))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(vocabulary_size, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()
    return model


model = create_model(vocabulary_size + 1, seq_len)
model.fit(X, y, batch_size=128, epochs=300, verbose=1)

from keras.preprocessing.sequence import pad_sequences


def generate_text(model, tokenizer, seq_len, seed_text, num_gen_words):
    output_text = []

    input_text = seed_text

    for i in range(num_gen_words):
        encoded_text = tokenizer.texts_to_sequences([input_text])[0]

        pad_encoded = pad_sequences([encoded_text], maxlen=seq_len, truncating='pre')
        pred_word_ind = model.predict_classes(pad_encoded, verbose=0)[0]
        pred_word = tokenizer.index_word[pred_word_ind]
        input_text += " " + pred_word
        output_text.append(pred_word)

    return ' '.join(output_text)


import random

random.seed(101)
random_pick = random.randint(0, len(sequences))

random_seed_text = text_sequence[random_pick]

seed_text = ' '.join(random_seed_text)

print(seed_text)
a = generate_text(model, tokenizer, seq_len, seed_text=seed_text, num_gen_words=25)

print(a)
