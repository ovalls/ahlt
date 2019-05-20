#! /usr/bin/python3

import sys
from os import listdir
import pickle
import gensim

from xml.dom.minidom import parse
from nltk.tokenize import word_tokenize

import numpy as np

import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, Conv1D, GlobalMaxPooling1D

from sklearn.metrics import r2_score

# --------- tokenize sentence -----------
# -- Tokenize sentence, returning tokens and span offsets

def tokenize(txt):
    offset = 0
    tks = []
    for t in word_tokenize(txt):
        offset = txt.find(t, offset)
        tks.append((t, offset, offset+len(t)-1))
        offset += len(t)
    return tks


# --------- MAIN PROGRAM -----------
# --
# -- Usage:  baseline-NER.py target-dir
# --
# -- Extracts Drug NE from all XML files in target-dir
# --


# directory with files to process
tr_datadir = sys.argv[1]
te_datadir = sys.argv[2]
#name_model = sys.argv[2]    # train or test

### Preparo per a CNN cada sentence, per a cada pair que troba
### M'és igual el nom de les drugs, el que vull és el contexte en el que estan
# Per a cada sentence tinc els tokens: tokens is a list of triples (word,start,end)
# - poso DrugX i DrugY en lloc dels noms de les entitats drogues (id_e1 i id_e2) i guardo en array (sentences_list)
# - poso class a la que pertanyen en un altre array (labels_list)
# cada sentence és una matriu pq. cada word és un vector d'unes 300 posicions que serà una row


###############################################################
################################ TRAINING SET #################
###############################################################

tr_sentences_list = []     # list of lists of tokens for the sentence of each pair -- for all the files
tr_temp_labels = []
tr_labels_list = []        # list of labels (class) for the sentence of each pair -- for all the files

# process each file in directory
for f in listdir(tr_datadir):

    # parse XML file, obtaining a DOM tree
    tree = parse(tr_datadir + "/" + f)

    # process each sentence in the file
    sentences = tree.getElementsByTagName("sentence")
    for s in sentences:
        sid = s.attributes["id"].value   # get sentence id
        stext = s.attributes["text"].value   # get sentence text

        tokens = tokenize(stext)

        # load sentence entities
        entities = {}
        ents = s.getElementsByTagName("entity")
        for e in ents:
           id = e.attributes["id"].value
           offs = e.attributes["charOffset"].value.split("-")
           txt = e.attributes["text"].value                     #### he afegit
           #entities[id] = offs
           entities[id] = txt                                   # m'interessa el text, no l'offset

        print('sentence: {}'.format(stext))

        # for each pair in the sentence, decide whether it is DDI and its type
        # només s'avaluen les sentences que tenen algun <pair>
        pairs = s.getElementsByTagName("pair")
        for p in pairs:
            id_e1 = p.attributes["e1"].value
            id_e2 = p.attributes["e2"].value
            is_ddi = p.attributes["ddi"].value
            if is_ddi == 'true':
                ddi_type = p.attributes["type"].value
            else:
                ddi_type = 'no-interaction'
            #(is_ddi,ddi_type) = check_interaction(tokens,entities,id_e1,id_e2)
            #ddi = "1" if is_ddi else "0"
            #print(sid+"|"+id_e1+"|"+id_e2+"|"+is_ddi+"|"+ddi_type)

            split_e1 = entities[id_e1].split()  # llista de les paraules que composes la drug (per si és composta)
            split_e2 = entities[id_e2].split()

            # each pair --> sentence = list of tokens --> drugs substituted by DrugX, DrugY
            tokens_list = []

            # per a cada token de la sentence de tokens
            len1, len2 = len(split_e1), len(split_e2)
            d1, d2 = '', ''
            found_d1, found_d2 = 0, 0
            check1, check2 = 1, 1

            for k in range(0, len(tokens)):
                t = tokens[k][0]
                #print('token_ids: {} / id_e1: {} / id_e2: {}'.format(tokens[k], id_e1, id_e2))
                #print('token_text: {} / id_e1: {} / id_e2: {}'.format(tokens[k], entities[id_e1], entities[id_e2]))
                #print('token: {}'.format(t))
                #print('split_e1: {}, len: {}'.format(split_e1, len(split_e1)))
                #print('split_e2: {}, len: {}'.format(split_e2, len(split_e2)))

                if t == entities[id_e1]:
                    tokens_list.append('DrugX')
                elif t == entities[id_e2]:
                    tokens_list.append('DrugY')
                elif len1 > 1 and check1 == 1:  #drug is more than one word long
                    #print('entra drug1')
                    if t in entities[id_e1]:
                        #print('DRUG1 starts with: {}'.format(t))
                        d1 = d1 + ' ' + t
                        found_d1 = 1
                    else:
                        if found_d1 == 1:
                            tokens_list.append('DrugX')
                            tokens_list.append(t)
                            check1 = 0
                        else:
                            #print('entra res de drug1')
                            tokens_list.append(t)
                elif len2 > 1 and check2 == 1:  #drug is more than one word long
                    #print('entra drug2')
                    if t in entities[id_e2]:
                        #print('DRUG2 starts with: {}'.format(t))
                        d2 = d2 + ' ' + t
                        found_d2 = 1
                    else:
                        if found_d2 == 1:
                            tokens_list.append('DrugY')
                            tokens_list.append(t)
                            check2 = 0
                        else:
                            #print('entra res de drug2')
                            tokens_list.append(t)
                else:
                    #print('entra res')
                    tokens_list.append(t)

            #print('Tokens PAIR:\n{}'.format(tokens_list))

            tr_sentences_list.append(tokens_list)
            tr_temp_labels.append(ddi_type)

tr_labels_list.append(tr_temp_labels)    # convertim llista de lables a llista de llistes (una) de labels

print('\n\n** Llista FINAL TRAIN **\n{}'.format(tr_sentences_list))
print('\n\n** Labels FINAL TRAIN **\n{}'.format(tr_labels_list))

'''
#Save lists to file
with open(name_model + '_sentences_list.data', 'wb') as file:
    # store the data as binary data stream
    pickle.dump(tr_sentences_list, file)
with open(name_model + '_labels_list.data', 'wb') as file:
    pickle.dump(tr_labels_list, file)
'''

# TRAIN Posem totes les paraules de totes les sentences a una llista
tr_words_list = []
for l in tr_sentences_list:
    for w in l:
        tr_words_list.append(w)

tr_unique_words = set(tr_words_list)
print('\n** unique words tr: {}'.format(tr_unique_words))
print('len: {}'.format(len(tr_unique_words)))

tr_words_list_labels = []
for l in tr_labels_list:
    for w in l:
        tr_words_list_labels.append(w)

tr_unique_words_labels = set(tr_words_list_labels)
print('\n** unique words labels tr: {}'.format(tr_unique_words_labels))
print('len: {}'.format(len(tr_unique_words_labels)))

# Dictionary of unique words with an index number assigned
#different words in the set
tr_dict_words = {ni: indi for indi, ni in enumerate(set(tr_words_list))}
tr_dict_words_labels = {ni: indi for indi, ni in enumerate(set(tr_words_list_labels))}
#numbers = [dict_words[ni] for ni in words_list]
#print('numbers: {}'.format(numbers))

#print(sentences_list[0][1])
#print(dict_words[sentences_list[0][1]])
#print(sentences_list[0][4])
#print(dict_words[sentences_list[0][4]])

# sentences_list convertida a indexos (índex assignat a cada paraula) -- categorical
for l in range(0, len(tr_sentences_list)):
    for w in range(0, len(tr_sentences_list[l])):
        tr_sentences_list[l][w] = tr_dict_words[tr_sentences_list[l][w]]

print('\n** indexes sentences_list tr: {}'.format(tr_sentences_list))

# labels_list convertida a indexos (índex assignat a cada paraula) [[3, 4, 5, 2, ...]] -- categorical
for l in range(0, len(tr_labels_list)):
    for w in range(0, len(tr_labels_list[l])):
        tr_labels_list[l][w] = tr_dict_words_labels[tr_labels_list[l][w]]

print('\n** indexes labels_list tr: {}'.format(tr_labels_list))

###############################################################
################################ TEST SET #####################
###############################################################

te_sentences_list = []     # list of lists of tokens for the sentence of each pair -- for all the files
te_temp_labels = []
te_labels_list = []        # list of labels (class) for the sentence of each pair -- for all the files
te_results = []

# process each file in directory
for f in listdir(te_datadir):

    # parse XML file, obtaining a DOM tree
    tree = parse(te_datadir + "/" + f)

    # process each sentence in the file
    sentences = tree.getElementsByTagName("sentence")
    for s in sentences:
        sid = s.attributes["id"].value   # get sentence id
        stext = s.attributes["text"].value   # get sentence text

        tokens = tokenize(stext)

        # load sentence entities
        entities = {}
        ents = s.getElementsByTagName("entity")
        for e in ents:
           id = e.attributes["id"].value
           offs = e.attributes["charOffset"].value.split("-")
           txt = e.attributes["text"].value
           entities[id] = txt

        print('sentence: {}'.format(stext))

        # for each pair in the sentence, decide whether it is DDI and its type
        # només s'avaluen les sentences que tenen algun <pair>
        pairs = s.getElementsByTagName("pair")
        for p in pairs:
            id_e1 = p.attributes["e1"].value
            id_e2 = p.attributes["e2"].value
            is_ddi = p.attributes["ddi"].value
            if is_ddi == 'true':
                ddi_type = p.attributes["type"].value
            else:
                ddi_type = 'no-interaction'
            print(sid+"|"+id_e1+"|"+id_e2+"|"+is_ddi+"|"+ddi_type)
            # Guardem a la llista de RESULTATS sid, id_e1 i id_e2, per després afegir les prediccions d'is_ddi i ddi_type
            te_results.append(sid+"|"+id_e1+"|"+id_e2+"|")

            split_e1 = entities[id_e1].split()  # llista de les paraules que composes la drug (per si és composta)
            split_e2 = entities[id_e2].split()

            # each pair --> sentence = list of tokens --> drugs substituted by DrugX, DrugY
            tokens_list = []

            # per a cada token de la sentence de tokens
            len1, len2 = len(split_e1), len(split_e2)
            d1, d2 = '', ''
            found_d1, found_d2 = 0, 0
            check1, check2 = 1, 1

            for k in range(0, len(tokens)):
                t = tokens[k][0]
                if t == entities[id_e1]:
                    tokens_list.append('DrugX')
                elif t == entities[id_e2]:
                    tokens_list.append('DrugY')
                elif len1 > 1 and check1 == 1:  #drug is more than one word long
                    if t in entities[id_e1]:
                        d1 = d1 + ' ' + t
                        found_d1 = 1
                    else:
                        if found_d1 == 1:
                            tokens_list.append('DrugX')
                            tokens_list.append(t)
                            check1 = 0
                        else:
                            tokens_list.append(t)
                elif len2 > 1 and check2 == 1:  #drug is more than one word long
                    if t in entities[id_e2]:
                        d2 = d2 + ' ' + t
                        found_d2 = 1
                    else:
                        if found_d2 == 1:
                            tokens_list.append('DrugY')
                            tokens_list.append(t)
                            check2 = 0
                        else:
                            tokens_list.append(t)
                else:
                    tokens_list.append(t)

            te_sentences_list.append(tokens_list)
            te_temp_labels.append(ddi_type)

te_labels_list.append(te_temp_labels)    # convertim llista de lables a llista de llistes (una) de labels

print('\n\n** Llista FINAL TEST **\n{}'.format(te_sentences_list))
print('\n\n** Labels FINAL TEST **\n{}'.format(te_labels_list))


# TEST Posem totes les paraules de totes les sentences a una llista
te_words_list = []
for l in te_sentences_list:
    for w in l:
        te_words_list.append(w)

te_unique_words = set(te_words_list)
print('\n** unique words te: {}'.format(te_unique_words))
print('len: {}'.format(len(te_unique_words)))

te_words_list_labels = []
for l in te_labels_list:
    for w in l:
        te_words_list_labels.append(w)

te_unique_words_labels = set(te_words_list_labels)
print('\n** unique words labels te: {}'.format(te_unique_words_labels))
print('len: {}'.format(len(te_unique_words_labels)))

# Dictionary of unique words with an index number assigned
#different words in the set
te_dict_words = {ni: indi for indi, ni in enumerate(set(te_words_list))}
te_dict_words_labels = {ni: indi for indi, ni in enumerate(set(te_words_list_labels))}
print('te_dict_words_labels: {}'.format(te_dict_words_labels))
#numbers = [te_dict_words_labels[ni] for ni in te_labels_list[0]]
#print('\n** numbers for labels: {}'.format(numbers))

# sentences_list convertida a indexos (índex assignat a cada paraula) -- categorical
for l in range(0, len(te_sentences_list)):
    for w in range(0, len(te_sentences_list[l])):
        te_sentences_list[l][w] = te_dict_words[te_sentences_list[l][w]]

print('\n** indexes sentences_list te: {}'.format(te_sentences_list))

# labels_list convertida a indexos (índex assignat a cada paraula) [[3, 4, 5, 2, ...]] -- categorical
for l in range(0, len(te_labels_list)):
    for w in range(0, len(te_labels_list[l])):
        te_labels_list[l][w] = te_dict_words_labels[te_labels_list[l][w]]

print('\n** indexes labels_list te: {}'.format(te_labels_list))

###############################################################
################################ LEARNING #####################
###############################################################

# Train and Test sets (all numpy arrays)
x_train = tr_sentences_list
y_train = np.asarray(tr_labels_list[0])         # Tinc [[3, 4, 3, 2, 1...]]: així trec llista fora
x_test = te_sentences_list
y_test = np.asarray(te_labels_list[0])

# ja els tinc categorical, no cal que posi keras.utils.to_categorical
num_classes = np.max(y_train) + 1       # (4 categories + no-categoria)
print('Num classes: {}'.format(num_classes))

max_words = len(tr_unique_words) + 1    # Keras: input_dim: int > 0. Size of the vocabulary, i.e. maximum integer index + 1.
embedding_dims = 300        # Keras: Dimension of the dense embedding.
# number of filters = number of output channels
filters = 256               # Keras: the dimensionality of the output space (i.e. the number of output filters in the convolution).
# original 250
kernel_size = 3             # Keras: length of the 1D convolution window.
# original 5

hidden_dims = 300
# original 150
batch_size = 64
epochs = 2

# max number of words in a sentence
maxlen_train = len(max(x_train,key=len))
maxlen_test = len(max(x_test,key=len))
maxlen = max(maxlen_train, maxlen_test)
print('maxlen tr vs te: {}'.format(maxlen))
maxlen = int(maxlen + 0.1*maxlen)   # afageixo 10% del màx, per si a test hi ha sentences + llarges
print('maxlen tr vs te + 10%: {}'.format(maxlen))

# Vectorize the output sentence type classifcations to Keras readable format
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Padding: Pad the input vectors to ensure a consistent length (181)
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

model = Sequential()

# Created Embedding (Input) Layer (max_words) --> Convolutional Layer
model.add(Embedding(max_words, embedding_dims, input_length=maxlen))
model.add(Dropout(0.2))  # masks various input values

print('\n ** Embedding Layer created **')

# Create the convolutional layer
model.add(Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1))

print('** Convolutional Layer created **')

# Create the pooling layer
model.add(GlobalMaxPooling1D())

print('** Pooling Layer created **')

# Create the fully connected layer
model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('relu'))

print('** Dense Layer ReLu created **')

# Create the output layer (num_classes)
model.add(Dense(num_classes))
model.add(Activation('softmax'))

print('** Dense Layer Softmax created **')

# Add optimization method, loss function and optimization value
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.compile(loss='categorical_crossentropy', optimizer='adam')

print('** Model compiled **')

'''
print('type, shape xtrain: {} / {}'.format(type(x_train),np.shape(x_train)))
print('type, shape ytrain: {} / {}'.format(type(x_train),np.shape(y_train)))
print('type, shape xtest: {} / {}'.format(type(x_train),np.shape(x_test)))
print('type, shape ytest: {} / {}'.format(type(x_train),np.shape(y_test)))

<class 'numpy.ndarray'>
<class 'numpy.ndarray'>
<class 'numpy.ndarray'>
<class 'numpy.ndarray'>
'''

# "Fit the model" (train model), using training data
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))

print('** Model fit **')
print('\nValidation loss (with test data): ', history.history['loss'])

# Evaluate the trained model, using the test data
score = model.evaluate(x_test, y_test, batch_size=batch_size)

print('\n** Score**\n{}'.format(score))

print('history.history:')
print(history.history)

'''

Y_pred = model.predict(x_test)
# Assign most probable label
y_pred = np.argmax(Y_pred, axis=1)

print('Y_pred: {}'.format(Y_pred))
print('\nPredicted labels (from test data):'.format(y_pred))
print('len Predicted labels (from test data):'.format(len(y_pred)))


test_yp = model.predict(x_test, batch_size=batch_size, verbose=0)
r2test = r2_score(y_test, test_yp)

print('y_predicted string: {}'.format(np.array2string(test_yp)))
print('y_predicted: {}'.format(test_yp))
print('r2_test: {}'.format(r2test))
'''

classes = model.predict_classes(x_test, batch_size=1)
print('classes: {}'.format(classes))

# te_dict_words_labels: {'no-interaction': 0, 'advise': 1, 'mechanism': 2, 'effect': 3, 'int': 4}
# reverse --> 0: no-interaction...
te_dict_words_labels_reverse = {indi: ni for indi, ni in enumerate(set(te_words_list_labels))}

te_dict_words_labels_more = []
for c in classes:
    print(c)
    # afegir resultats a la llista te_results
    if c == te_dict_words_labels['no-interaction']:
        is_ddi = 'false'
        ddi_type = 'no-interaction'
    else:
        is_ddi = 'true'
        ddi_type = te_dict_words_labels_reverse[c]
    te_dict_words_labels_more.append(is_ddi + "|" + ddi_type)

#print(sid + "|" + id_e1 + "|" + id_e2 + "|" + is_ddi + "|" + ddi_type)
#te_results.append(sid + "|" + id_e1 + "|" + id_e2 + "|")

print('\nte_dict_words_labels_more\n{}'.format(te_dict_words_labels_more))

for t in range(0, len(te_dict_words_labels_more)):
    te_results[t] = te_results[t] + te_dict_words_labels_more[t]

print('\n\n** te_results FINAL\n{}'.format(te_results))
for t in te_results:
    print(t)

