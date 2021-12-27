import pickle
import numpy as np

with open("train_qa.txt", "rb") as f:
    train_data = pickle.load(f)

with open("test_qa.txt", "rb") as f:
    text_data = pickle.load(f)

all_data = train_data + text_data

vocab = set()

for story, question, answer in all_data:
    vocab = vocab.union(set(story))
    vocab = vocab.union(set(question))

vocab.add("yes")
vocab.add("no")

vocab_len = len(vocab) + 1

all_story_lens = [len(data[0]) for data in all_data]
max_story_len = max(all_story_lens)
max_question_len = max([len(data[1]) for data in all_data])

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(filters=[])
tokenizer.fit_on_texts(vocab)

print(tokenizer.word_index)

train_story_text = []
train_question_text = []
train_answer = []

for story, question, answer in train_data:
    train_story_text.append(story)
    train_question_text.append(question)
    train_answer.append(answer)

train_story_seq = tokenizer.texts_to_sequences(train_story_text)


def vectorize_stories(data, word_index=tokenizer.word_index, max_story_len=max_story_len,
                      max_question_len=max_question_len):
    # vectorized stories:
    X = []
    # vectorized questions:
    Xq = []
    # vectorized answers:
    Y = []

    for story, question, answer in data:
        # Getting indexes for each word in the story
        x = [word_index[word.lower()] for word in story]
        # Getting indexes for each word in the story
        xq = [word_index[word.lower()] for word in question]
        # For the answers
        y = np.zeros(len(word_index) + 1)  # Index 0 Reserved when padding the sequences
        y[word_index[answer]] = 1

        X.append(x)
        Xq.append(xq)
        Y.append(y)

    # Now we have to pad these sequences:
    return (pad_sequences(X, maxlen=max_story_len), pad_sequences(Xq, maxlen=max_question_len), np.array(Y))


input_trains, queries_trains, answers_train = vectorize_stories(train_data)
input_test, queries_test, answers_test = vectorize_stories(text_data)
print(input_test)
sum(answers_test)

from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import Input, Activation, Dense, Permute, Dropout, add, dot, concatenate, LSTM

input_sequence = Input((max_story_len,))
question = Input((max_question_len,))

vocab_size = len(vocab) + 1

input_encoded_m = Sequential()
input_encoded_m.add(Embedding(input_dim=vocab_size, output_dim=64))
input_encoded_m.add(Dropout(0.3))

input_encoded_c = Sequential()
input_encoded_c.add(Embedding(input_dim=vocab_size, output_dim=max_question_len))
input_encoded_c.add(Dropout(0.3))

question_encoder = Sequential()
question_encoder.add(Embedding(input_dim=vocab_size, output_dim=64, input_length=max_question_len))
question_encoder.add(Dropout(0.3))

input_enc_m = input_encoded_m(input_sequence)
input_enc_c = input_encoded_c(input_sequence)

question_encoded = question_encoder(question)

match = dot([input_enc_m, question_encoded], axes=(2, 2))
match = Activation("softmax")(match)

response = add([match, input_enc_c])
response = Permute((2, 1))(response)

answer = concatenate([response, question_encoded])
answer = LSTM(32)(answer)

answer = Dropout(0.5)(answer)
answer = Dense(vocab_size)(answer)

answer = Activation("softmax")(answer)

model = Model([input_sequence, question], answer)

model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

history = model.fit([input_trains, queries_trains], answers_train, batch_size=32, epochs=3,
                    validation_data=([input_test, queries_test], answers_test))
print("----------------------")

my_story = "Sandra picked up the milk . Mary travelled left . "
my_story.split()

my_question = 'Sandra got the milk ?'
my_question.split()

my_data = [(my_story.split(), my_question.split(), 'no')]

my_story, my_ques, my_ans = vectorize_stories(my_data)

pred_results = model.predict(([my_story, my_ques]))
val_max = np.argmax(pred_results[0])
# Correct prediction!
for key, val in tokenizer.word_index.items():
    if val == val_max:
        k = key
print(k)
# print(pred_results[0][val_max])
