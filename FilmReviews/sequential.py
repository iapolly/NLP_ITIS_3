import pymorphy2 as pymorphy2
from keras.models import Sequential
from keras.layers import Dense, Dropout
import pandas as pd
from nltk import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import keras

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

tf_idf_1000 = TfidfVectorizer(max_features=1000)

symbol = " "
all_feeds = [symbol.join(entry) for entry in all_feeds]
my_feeds = [symbol.join(entry) for entry in my_feeds]

X_train = tf_idf_1000.fit_transform(all_feeds)
x_test = tf_idf_1000.fit_transform(my_feeds)

X_train = get_normal_form_words(all_feeds)
x_test = get_normal_form_words(my_feeds)

Y_train = keras.utils.to_categorical(all_labels, 3)
y_test = keras.utils.to_categorical(my_labels, 3)

model = Sequential()
model.add(Dense(512, input_shape=(1000,)))
model.add(Dropout(0.5))
model.add(Dense(3))
model.compile(metrics=["accuracy"], optimizer='adam', loss='categorical_crossentropy')

model.fit(X_train, Y_train, epochs=10, batch_size=1000)

y_pred = model.predict(x_test)

print(classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1)))
