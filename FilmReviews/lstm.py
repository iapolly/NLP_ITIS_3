import keras
import pandas as pd
import numpy as np
import pymorphy2
from keras import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPool1D, Activation, Dense, LSTM

from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

all_feeds_file = "all_train.csv"
my_feeds_file = "my.csv"

morph = pymorphy2.MorphAnalyzer()

all_feeds = pd.read_csv(
    all_feeds_file,
    encoding="utf-8",
    usecols=["text"]
).values

all_labels = pd.read_csv(
    all_feeds_file,
    encoding="utf-8",
    usecols=["label"]
).values

my_feeds = pd.read_csv(
    my_feeds_file,
    encoding="utf-8",
    usecols=["text"]
).values

my_labels = pd.read_csv(
    my_feeds_file,
    encoding="utf-8",
    usecols=["label"]
).values

Y_train = []
for label in all_labels:
    Y_train.append(int(label[0]))

y_true = []
for label in my_labels:
    y_true.append(int(label[0]))


def create_embedding_matrix(model, word_index, embedding_dim):
    vocab_size = len(word_index) + 1

    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for index in range(len(model["word"])):
        if model["word"][index] in word_index:
            idx = word_index[model["word"][index]]
            embedding_matrix[idx] = np.array(
                model["value"][index], dtype=np.float32)[:embedding_dim]

    return embedding_matrix


def read_from_pickle(name):
    return pd.read_pickle(name)


words_count = 189193
vector_size = 300
categories_count = 3
sequence_size = 1000

filename = 'file.pkl'
model = read_from_pickle(filename)

tokenizer = Tokenizer(num_words=words_count)
tokenizer.fit_on_texts(model["word"])

symbol = " "
all_feeds = [symbol.join(entry) for entry in all_feeds]
my_feeds = [symbol.join(entry) for entry in my_feeds]

X_train = tokenizer.texts_to_sequences(all_feeds)
x_test = tokenizer.texts_to_sequences(my_feeds)

X_train = pad_sequences(X_train, maxlen=sequence_size, padding='post')
x_test = pad_sequences(x_test, maxlen=sequence_size, padding='post')
Y_train = keras.utils.to_categorical(Y_train, categories_count)
y_test = keras.utils.to_categorical(y_true, categories_count)

embedding_matrix = create_embedding_matrix(model, tokenizer.word_index, vector_size)

vocab_size = len(tokenizer.word_index.keys()) + 1

model = Sequential()
model.add(Embedding(vocab_size, vector_size, weights=[embedding_matrix], input_length=sequence_size))
model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(3, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

batch_size = 32
epochs = 10

model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)

print()
print(u'Оценка теста: {}'.format(score[0]))
print(u'Оценка точности модели: {}'.format(score[1]))

result = model.predict(x_test)
print(classification_report(y_test.argmax(axis=1), result.argmax(axis=1)))