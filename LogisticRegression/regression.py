import numpy as np
import pandas as pd
import pymorphy2
from nltk import RegexpTokenizer
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import accuracy_score

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


all_unique_words = np.unique(get_normal_form_words(all_feeds))

# words_bag = get_words_bag(all_feeds, all_unique_words)
# my_words_bag = get_words_bag(my_feeds, all_unique_words)

# save_array_to_file(words_bag, "all_bag.txt")
# save_array_to_file(my_words_bag, "my_bag.txt")

words_bag = read_array_from_file("all_bag.txt")
my_words_bag = read_array_from_file("my_bag.txt")

reg_model = LogisticRegression(max_iter=100000)
reg_model.fit(words_bag, Y_train)

y_pred = reg_model.predict(my_words_bag)

# метрики
print(accuracy_score(y_true, y_pred))
print(metrics.classification_report(y_true, y_pred, digits=3))

positive_weights = dict(zip(all_unique_words, reg_model.coef_[2]))
neutral_weights = dict(zip(all_unique_words, reg_model.coef_[1]))
negative_weights = dict(zip(all_unique_words, reg_model.coef_[0]))


def get_first_last_words(dict_weights):
    sorted_list = list({k: v for k, v in sorted(dict_weights.items(), key=lambda item: item[1], reverse=True)})
    first_10 = sorted_list[0:10]
    reversed_sorted_list = sorted_list[::-1]
    last_10 = reversed_sorted_list[0:10]
    print(first_10, last_10)


# первые 10 и последние 10 слов для каждого класса
get_first_last_words(positive_weights)
get_first_last_words(negative_weights)
get_first_last_words(negative_weights)
