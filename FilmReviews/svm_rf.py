import string

import numpy as np
import pandas as pd
import pymorphy2
from nltk import RegexpTokenizer
import pickle

from sklearn import svm
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

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


def get_normal_form(one_word):
    return morph.parse(one_word)[0].normal_form


tokenizer = RegexpTokenizer(r'\w+')


# массив всех слов в нормальной форме
def get_normal_form_words(feeds):
    normal_form_words = []
    for feed in feeds:
        words = tokenizer.tokenize(feed[0].lower())
        normal_form_words += list(map(get_normal_form, words))

    return normal_form_words


# количество уникальных слов в массиве
def get_count_of_unique_words(words_array):
    return len(np.unique(words_array))


# мешок слов
def get_words_bag(feeds, all_unique):
    bag = []
    for feed in feeds:
        feed_all_words = get_normal_form_words([feed])
        feed_bag = []
        for word in all_unique:
            count = feed_all_words.count(word)
            feed_bag.append(count)

        bag.append(feed_bag)

    return bag


def save_array_to_file(array, text):
    with open(text, 'wb') as f:
        pickle.dump(array, f)


def read_array_from_file(text):
    with open(text, 'rb') as f:
        return pickle.load(f)


analyzer = pymorphy2.MorphAnalyzer()


def get_parts_of_speech(texts):
    parts_vector = []
    for text in texts:
        nouns_count = 0
        verbs_count = 0
        adjectives_count = 0
        adverbs_count = 0
        for word in text:
            word = analyzer.parse(word)[0].tag.POS
            if word == 'NOUN':
                nouns_count += 1
            elif word == 'VERB' or word == 'INFN':
                verbs_count += 1
            elif word == 'ADJF' or word == 'COMP' or word == 'ADJS':
                adjectives_count += 1
            elif word == 'ADVB':
                adverbs_count += 1
        parts_vector.append([nouns_count, verbs_count, adjectives_count, adverbs_count])
    return parts_vector


def get_punctuation_count(texts):
    punctuation_vector = []
    for text in texts:
        result_row = []
        for punctuation in string.punctuation:
            single_punctuation_counter = 0
            for word in text:
                if word == punctuation:
                    single_punctuation_counter += 1
            result_row.append(single_punctuation_counter)
        punctuation_vector.append(result_row)
    return punctuation_vector


all_unique_words = np.unique(get_normal_form_words(all_feeds))

# words_bag = get_words_bag(all_feeds, all_unique_words)
# my_words_bag = get_words_bag(my_feeds, all_unique_words)

# save_array_to_file(words_bag, "all_bag.txt")
# save_array_to_file(my_words_bag, "my_bag.txt")

words_bag = read_array_from_file("all_bag.txt")
my_words_bag = read_array_from_file("my_bag.txt")


# svm_model = svm.LinearSVC(max_iter=100000)
# svm_model.fit(words_bag, Y_train)
#
# y_pred = svm_model.predict(my_words_bag)

def append_vectors(vector0, vector1):
    idx = 0
    for v0 in vector0:
        for v1 in vector1[idx]:
            v0.append(v1)
        idx += 1


parts_train = get_parts_of_speech(all_feeds)
parts_test = get_parts_of_speech(my_feeds)

punct_train = get_punctuation_count(all_feeds)
punct_test = get_punctuation_count(my_feeds)

# bow + pos + puct
append_vectors(words_bag, parts_train)
append_vectors(words_bag, punct_train)

append_vectors(my_words_bag, parts_test)
append_vectors(my_words_bag, punct_test)

# bow + pos
# append_vectors(words_bag, parts_train)
# append_vectors(my_words_bag, parts_test)

# bow + puct
# append_vectors(words_bag, punct_train)
# append_vectors(my_words_bag, punct_test)

# pos + puct
# append_vectors(parts_train, punct_train)
# append_vectors(parts_test, punct_test)

rfc_model = RandomForestClassifier(max_depth=20)
rfc_model.fit(parts_train, Y_train)

y_pred = rfc_model.predict(parts_test)

print(metrics.classification_report(y_true, y_pred, digits=3))