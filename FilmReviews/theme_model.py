from multiprocessing import freeze_support

import gensim
import pandas as pd
import pymorphy2
from gensim import corpora
from gensim.models import CoherenceModel
from nltk import RegexpTokenizer

all_feeds_file = "all_train.csv"
my_feeds_file = "my.csv"

morph = pymorphy2.MorphAnalyzer()

all_feeds = pd.read_csv(
    all_feeds_file,
    encoding="utf-8",
    usecols=["text"]
).values

my_feeds = pd.read_csv(
    my_feeds_file,
    encoding="utf-8",
    usecols=["text"]
).values


def save_to_pickle(df, name):
    pd.to_pickle(df, name)


def read_from_pickle(name):
    return pd.read_pickle(name)


def get_normal_form(one_word):
    return morph.parse(one_word)[0].normal_form


tokenizer = RegexpTokenizer(r'\w+')


# массивы всех слов в нормальной форме
def get_normal_form_words(feeds):
    normal_form_words = []
    for feed in feeds:
        words = tokenizer.tokenize(feed[0].lower())
        normal_form_words.append(list(map(get_normal_form, words)))

    return normal_form_words


if __name__ == '__main__':
    normalized = read_from_pickle('all_norm.pkl')

    dictionary = corpora.Dictionary(normalized)
    corpus = [dictionary.doc2bow(text) for text in normalized]

    NUM_TOPICS = 10
    model = gensim.models.ldamodel.LdaModel(corpus, num_topics=NUM_TOPICS, id2word=dictionary, passes=15)

    NUM_WORDS = 15
    topics = model.print_topics(num_words=NUM_WORDS)
    for topic in topics:
        print(topic)

    coh_model = CoherenceModel(model=model, texts=normalized, dictionary=dictionary)
    coherence = coh_model.get_coherence()
    print(coherence)