#! /usr/bin/python3

import sys
from os import listdir

from xml.dom.minidom import parse
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.wsd import lesk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn
from urllib.request import urlretrieve
from sh import gunzip
import gensim
from sklearn.naive_bayes import GaussianNB

import numpy as np


def process_set(datadir):
    set_sentences = {}
    set_entities = {}
    set_pairs = {}

    # process each file in directory
    for f in listdir(datadir):
        # parse XML file, obtaining a DOM tree
        tree = parse(datadir + "/" + f)

        # process each sentence in the file
        sentences = tree.getElementsByTagName("sentence")
        for s in sentences:
            sid = s.attributes["id"].value   # get sentence id
            stext = s.attributes["text"].value   # get sentence text
            set_sentences[sid] = stext

            # load sentence entities
            ents = s.getElementsByTagName("entity")
            for e in ents:
               id = e.attributes["id"].value
               offs = e.attributes["charOffset"].value.split("-")
               txt = e.attributes["text"].value
               #entities[id] = offs
               set_entities[id] = txt

            # for each pair in the sentence, decide whether it is DDI and its type
            pairs = s.getElementsByTagName("pair")
            for p in pairs:
                id_e1 = p.attributes["e1"].value
                id_e2 = p.attributes["e2"].value
                is_ddi = p.attributes["ddi"].value
                ddi_type = "null" if is_ddi != 'true' else p.attributes["type"].value
                pair_id = p.attributes["id"].value

                set_pairs[pair_id] = (sid, id_e1, id_e2, ddi_type)

    return set_sentences, set_entities, set_pairs


def get_verb(tags, d1, d2):
    verbs = []
    for i, t in enumerate(tags):
        if t[1][0] == 'V':
            if t[0].lower() != tags[d1][0].lower() and t[0].lower() != tags[d2][0].lower():
                if i == len(tags) - 1 or tags[i+1][1][0] != 'V':
                    distance = min(abs(i-d1),abs(i-d2))
                    verbs.append((t[0],distance))
    verbs.sort(key=lambda x: x[1])
    return verbs


def get_index(tokens, element):
    ind = [i for i, s in enumerate(tokens) if element.lower() in s.lower() or s.lower() in element.lower()]
    return ind[0] if len(ind) > 0 else -1


def penn2morphy(penntag, returnNone=False):
    morphy_tag = {'NN':wn.NOUN, 'JJ':wn.ADJ,
                  'VB':wn.VERB, 'RB':wn.ADV}
    try:
        return morphy_tag[penntag[:2]]
    except:
        return None if returnNone else ''


def get_word_embedding(w):
    if w.lower() in model:
        return model[w.lower()]
    else:
        return np.zeros(len(model[model.index2word[0]]))


def get_sentence_embedding(tokens):
    embeddings = []
    for w in tokens:
        embeddings.append(get_word_embedding(w))
    if len(embeddings) > 1:
        return np.mean(embeddings, axis=0)
    elif len(embeddings) == 1:
        return embeddings[0]
    else:
        return np.zeros(len(model[model.index2word[0]]))


def extract_features(sentences, entities, pairs):
    features = []
    indices = []
    labels = []

    for p in pairs:
        tokens = [w for w in word_tokenize(sentences[pairs[p][0]]) if w not in stop_words]
        sent_emb = get_sentence_embedding(tokens)
        tags = nltk.pos_tag(tokens)
        d1 = get_index(tokens, entities[pairs[p][1]])
        drug1 = entities[pairs[p][1]].replace(' ','-')
        d2 = get_index(tokens, entities[pairs[p][2]])
        drug2 = entities[pairs[p][2]].replace(' ','-')
        verbs = get_verb(tags,d1,d2)
        main_verb = lemma.lemmatize(verbs[0][0], 'v') if len(verbs) > 0 else 'none'
        word_emb = get_word_embedding(main_verb)
        features.append([word_emb, sent_emb])
        indices.append(p)
        labels.append(pairs[p][-1])

    features = np.array(features)
    nsamples, nx, ny = features.shape
    features = features.reshape((nsamples,nx*ny))
    return features, indices, labels


# --------- MAIN PROGRAM -----------
# --
# -- Usage:
# --


# directory with files to process
train_datadir = sys.argv[1]
test_datadir = sys.argv[2]

# nltk.download('stopwords')

# If it does not work, download, unzip and rename the following file to model.bin
url = "https://github.com/eyaler/word2vec-slim/raw/master/GoogleNews-vectors-negative300-SLIM.bin.gz"
urlretrieve(url, "model.bin.gz")
gunzip("model.bin.gz")
model = gensim.models.KeyedVectors.load_word2vec_format('model.bin', binary=True)

lemma = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

nb = GaussianNB()

train_sentences, train_entities, train_pairs = process_set(train_datadir)
train_features, train_indices, train_labels = extract_features(train_sentences, train_entities, train_pairs)

nb.fit(train_features, train_labels)

test_sentences, test_entities, test_pairs = process_set(test_datadir)
test_features, test_indices, test_labels = extract_features(test_sentences, test_entities, test_pairs)

predictions = nb.predict(test_features).tolist()

for i, p in enumerate(predictions):
    value = '0' if p == 'null' else '1'
    line = test_pairs[test_indices[i]][0]+'|'+test_pairs[test_indices[i]][1]+'|'+test_pairs[test_indices[i]][2]+'|'+value+'|'+p
    print(line)
