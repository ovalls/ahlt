#! /usr/bin/python3

import sys
from os import listdir

from xml.dom.minidom import parse
from nltk.tokenize import word_tokenize

import multiprocessing
from gensim.models import Word2Vec
import gensim
from time import time  # To time our operations

import numpy as np

import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, Conv1D, GlobalMaxPooling1D, MaxPooling1D, Flatten

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
# -- Usage:
# --


# directory with files to process
tr_datadir = sys.argv[1]
te_datadir = sys.argv[2]
te_resultsdir = sys.argv[3]

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
           txt = e.attributes["text"].value
           #entities[id] = offs
           entities[id] = txt                                   # m'interessa el text, no l'offset

        #print('sentence: {}'.format(stext))

        # for each pair in the sentence, decide whether it is DDI and its type
        # només s'avaluen les sentences que tenen algun <pair>
        pairs = s.getElementsByTagName("pair")
        for p in pairs:
            id_e1 = p.attributes["e1"].value
            id_e2 = p.attributes["e2"].value
            is_ddi = p.attributes["ddi"].value
            if is_ddi == 'true':                            # Crec que a train no necessito aquest IF !!!!!
                ddi_type = p.attributes["type"].value if p.hasAttribute("type") else "effect"
            else:
                ddi_type = 'null'
            #(is_ddi,ddi_type) = check_interaction(tokens,entities,id_e1,id_e2)
            #ddi = "1" if is_ddi else "0"
            #print(sid+"|"+id_e1+"|"+id_e2+"|"+is_ddi+"|"+ddi_type)
            # Aquí no necessitem guardar sid, id_e1 i id_e2 pq. no predim is_ddi i ddi_type.
            # Amb train només entrenem. A test sí que guardarem el resultat que predim

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

#print('\n\n** Llista FINAL TRAIN **\n{}'.format(tr_sentences_list))
#print('\n\n** Labels FINAL TRAIN **\n{}'.format(tr_labels_list))

print('\ntr_sentences_list[4]: {}'.format(tr_sentences_list[4]))

#################################### DICCIONARIS ############################################
UNK = '<unk>'
idx2symbol = list(set([word for sentence in tr_sentences_list for word in sentence] + [UNK]))
symbol2idx = {symbol: idx for idx, symbol in enumerate(idx2symbol)}
UNK_IDX = symbol2idx[UNK]

idx2label =  list(set([label for labels in tr_labels_list for label in labels] + [UNK]))
label2idx = {label: idx for idx, label in enumerate(idx2label)}

#################################### WORDS a INTS ################################################

# sentences_list convertida a indexos (índex assignat a cada paraula) -- categorical
for l in range(0, len(tr_sentences_list)):
    for w in range(0, len(tr_sentences_list[l])):
        tr_sentences_list[l][w] = symbol2idx.get(tr_sentences_list[l][w], UNK_IDX)

#print('\n** indexes sentences_list tr: {}'.format(tr_sentences_list))

# labels_list convertida a indexos (índex assignat a cada paraula) [[3, 4, 5, 2, ...]] -- categorical
for l in range(0, len(tr_labels_list)):
    for w in range(0, len(tr_labels_list[l])):
        tr_labels_list[l][w] = label2idx[tr_labels_list[l][w]]

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

        #print('sentence: {}'.format(stext))

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
                ddi_type = 'null'
            #print(sid+"|"+id_e1+"|"+id_e2+"|"+is_ddi+"|"+ddi_type)
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

#print('\n\n** Llista FINAL TEST **\n{}'.format(te_sentences_list))
#print('\n\n** Labels FINAL TEST **\n{}'.format(te_labels_list))


#################################### WORDS a INTS ################################################

# sentences_list convertida a indexos (índex assignat a cada paraula) -- categorical
for l in range(0, len(te_sentences_list)):
    for w in range(0, len(te_sentences_list[l])):
        te_sentences_list[l][w] = symbol2idx.get(te_sentences_list[l][w], UNK_IDX)

#print('\n** indexes sentences_list te: {}'.format(te_sentences_list))

# labels_list convertida a indexos (índex assignat a cada paraula) [[3, 4, 5, 2, ...]] -- categorical
for l in range(0, len(te_labels_list)):
    for w in range(0, len(te_labels_list[l])):
        te_labels_list[l][w] = label2idx[te_labels_list[l][w]]

#print('\n** indexes labels_list te: {}'.format(te_labels_list))


############################################################################################
################################ LEARNING Convolutional Neural Network #####################
############################################################################################

# Train and Test sets (all numpy arrays)
x_train = tr_sentences_list
y_train = np.asarray(tr_labels_list[0])
x_test = te_sentences_list
y_test = np.asarray(te_labels_list[0])

# Ratio of each class in the training data
for v in range(len(idx2label)):
  print("{}: {}".format(v, (y_train == v).sum()/y_train.shape[0]))

class_counts = {v: (y_train == v).sum() for v in range(len(idx2label))}
max_count = max(class_counts.values())
class_weights = {v: float(max_count)/class_counts[v] for v in range(len(idx2label))}


### A veure què tinc dins dels arrays ###
print('\n** ---------- DINS ARRAYS ----------- **')
print('* x_train[0]\n {}\n* x_train[100]\n {}'.format(x_train[0], x_train[100]))
print('* y_train[0]\n {}\n* y_train[100]\n {}'.format(y_train[0], y_train[100]))
print('* x_test[0]\n {}\n* x_test[100]\n {}'.format(x_test[0], x_test[100]))
print('* y_test[0]\n {}\n* y_test[100]\n {}'.format(y_test[0], y_test[100]))
print('** -------- FI DINS ARRAYS ---------- **')


num_classes = np.max(y_train) + 1       # (4 categories + no-categoria)            ###
print('Num classes: {}'.format(num_classes))

max_words = len(idx2symbol)    # size of vocabulary
embedding_dims = 300        # Keras: Dimension of the dense embedding.
# number of filters = number of output channels
#filters = 256               # Keras: the dimensionality of the output space (i.e. the number of output filters in the convolution).
filters = 128
# original 250
#kernel_size = 3             # Keras: length of the 1D convolution window.
kernel_size = 5 # original

hidden_dims = 128
# original 150
batch_size = 64
#batch_size = 128
epochs = 10


# max number of words in a sentence
maxlen_train = len(max(x_train,key=len))
maxlen_test = len(max(x_test,key=len))
maxlen = max(maxlen_train, maxlen_test)
#print('maxlen tr vs te: {}'.format(maxlen))
maxlen = int(maxlen + 0.1*maxlen)   # afageixo 10% del màx, per si a test hi ha sentences + llarges
#print('maxlen tr vs te + 10%: {}'.format(maxlen))

# Vectorize the output sentence type classifcations to Keras readable format
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Padding: Pad the input vectors to ensure a consistent length (181)
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

model = Sequential()

# Created Embedding (Input) Layer (max_words) --> Convolutional Layer
model.add(Embedding(max_words, embedding_dims, input_length=maxlen))

# weights from pre-trained embeddings
#model.add(Embedding(max_words, embedding_dims, weights=[embedding_matrix], input_length=maxlen))
# weights from Glove embeddings
#model.add(Embedding(max_words, embedding_dims, weights=[embedding_matrix_glove], input_length=maxlen))
#model.add(Dropout(0.2))  # masks various input values

print('\n ** Embedding Layer created **')

# Create the convolutional layer
model.add(Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1))
#model.add(Conv1D(128, 5, padding='valid', activation='relu'))

print('** Convolutional Layer created **')

# Create the pooling layer
model.add(GlobalMaxPooling1D())
#model.add(MaxPooling1D(5))

print('** Pooling Layer created **')

# Create the fully connected layer
model.add(Dense(hidden_dims))

#model.add(Dropout(0.2))
model.add(Activation('relu'))

print('** Dense Layer ReLu created **')

# Create the output layer (num_classes)
model.add(Dense(num_classes))
model.add(Activation('softmax'))

print('** Dense Layer Softmax created **')

# Add optimization method, loss function and optimization value

optimizer = keras.optimizers.Adam(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.summary()

print('** Model compiled **')

# "Fit the model" (train model), using training data
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test)) # , class_weight=class_weights)

print('** Model fit **')
#print('\nValidation loss (with test data): ', history.history['loss'])

# Evaluate the trained model, using the test data
#score = model.evaluate(x_test, y_test, batch_size=batch_size)
#print('\n** Score**\n{}'.format(score))

loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print('Accuracy: %f' % (accuracy*100))


print('history.history:\n{}'.format(history.history))

classes = model.predict_classes(x_test, batch_size=1)


print('\n** Classes: {}'.format(classes))

# te_dict_words_labels: {'null': 0, 'advise': 1, 'mechanism': 2, 'effect': 3, 'int': 4}
# reverse --> 0: null...

te_dict_words_labels_more = []
NULL_CLASS = label2idx['null']
for c in classes:
    #print(c)
    # afegir resultats a la llista te_results
    is_ddi = '0' if c == NULL_CLASS else '1'
    ddi_type = idx2label[c]
    te_dict_words_labels_more.append(is_ddi + "|" + ddi_type)

#print(sid + "|" + id_e1 + "|" + id_e2 + "|" + is_ddi + "|" + ddi_type)
#te_results.append(sid + "|" + id_e1 + "|" + id_e2 + "|")

#print('\nte_dict_words_labels_more\n{}'.format(te_dict_words_labels_more))

for t in range(0, len(te_dict_words_labels_more)):
    te_results[t] = te_results[t] + te_dict_words_labels_more[t]

#print('\n\n** te_results FINAL\n{}'.format(te_results))

for t in te_results:
    print(t)

with open(te_resultsdir, "w") as file:
    for t in te_results:
        file.write(t)
        file.write("\n")

#### AFEGIR
# dalt: Usage: xxxx com s'executa el programa i què fa


