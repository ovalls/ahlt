#! /usr/bin/python3

import sys
from os import listdir
import pickle
import gensim

from xml.dom.minidom import parse
from nltk.tokenize import word_tokenize

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
            print(sid+"|"+id_e1+"|"+id_e2+"|"+is_ddi+"|"+ddi_type)

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

# sentences_list convertida a indexos (índex assignat a cada paraula)
for l in range(0, len(tr_sentences_list)):
    for w in range(0, len(tr_sentences_list[l])):
        tr_sentences_list[l][w] = tr_dict_words[tr_sentences_list[l][w]]

print('\n** indexes sentences_list tr: {}'.format(tr_sentences_list))

# labels_list convertida a indexos (índex assignat a cada paraula) [[3, 4, 5, 2, ...]]
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

# sentences_list convertida a indexos (índex assignat a cada paraula)
for l in range(0, len(te_sentences_list)):
    for w in range(0, len(te_sentences_list[l])):
        te_sentences_list[l][w] = te_dict_words[te_sentences_list[l][w]]

print('\n** indexes sentences_list te: {}'.format(te_sentences_list))

# labels_list convertida a indexos (índex assignat a cada paraula) [[3, 4, 5, 2, ...]]
for l in range(0, len(te_labels_list)):
    for w in range(0, len(te_labels_list[l])):
        te_labels_list[l][w] = te_dict_words_labels[te_labels_list[l][w]]

print('\n** indexes labels_list te: {}'.format(te_labels_list))
