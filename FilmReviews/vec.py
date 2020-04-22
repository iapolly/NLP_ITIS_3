import gensim
import pymorphy2 as pymorphy2
import pandas as pd
from nltk import RegexpTokenizer
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


# массив массивов всех слов в нормальной форме
def get_normal_form_words(feeds):
    normal_form_words = []
    for feed in feeds:
        words = tokenizer.tokenize(feed[0].lower())
        normal_form_words.append(list(map(get_normal_form, words)))

    return normal_form_words


nf_feeds = get_normal_form_words(all_feeds)
nf_test = get_normal_form_words(my_feeds)

model = gensim.models.Word2Vec(nf_feeds, min_count=0)

print(nf_feeds)
print(model.wv.most_similar('кино'))
print(model.wv.most_similar('актёр'))
print(model.wv.most_similar('красота'))
print(model.wv.most_similar('америка'))
print(model.wv.most_similar('фильм'))


def get_median_vectors(arrays, size):
    result = [0] * size

    for array in arrays:
        for j in range(len(array)):
            result[j] += array[j]

    idx = 0
    for value in result:
        result[idx] = value/len(arrays)
    return result


def get_vectors(reviews, size, is_test):
    result = pd.DataFrame(columns=[0] * size)

    idx = 0
    for review in reviews:
        idx += 1
        arrays = []
        for word in review:
            if word in list(model.wv.vocab.keys()) or not is_test:
                vector = model.wv[word]
                arrays.append(vector)
            else:
                vector = []
                for i in range(size):
                    vector.append(0)

                arrays.append(vector)
        result.loc[idx] = get_median_vectors(arrays, size)
    return result


SIZE = 300
x_train = get_vectors(nf_feeds, size=SIZE, is_test=False)
x_test = get_vectors(nf_test, size=SIZE, is_test=True)

rfc_model = RandomForestClassifier(max_depth=20)
rfc_model.fit(x_train, Y_train)
y_pred = rfc_model.predict(x_test)

print(metrics.classification_report(y_true, y_pred, digits=3))
