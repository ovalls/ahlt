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
datadir = sys.argv[1]
name_model = sys.argv[2]    # train or test

### Preparo per a CNN cada sentence, per a cada pair que troba
### M'és igual el nom de les drugs, el que vull és el contexte en el que estan
# Per a cada sentence tinc els tokens: tokens is a list of triples (word,start,end)
# - poso DrugX i DrugY en lloc dels noms de les entitats drogues (id_e1 i id_e2) i guardo en array (sentences_list)
# - poso class a la que pertanyen en un altre array (labels_list)
# cada sentence és una matriu pq. cada word és un vector d'unes 300 posicions que serà una row

sentences_list = []     # list of lists of tokens for the sentence of each pair -- for all the files
temp_labels = []
labels_list = []        # list of labels (class) for the sentence of each pair -- for all the files

# process each file in directory
for f in listdir(datadir):

    # parse XML file, obtaining a DOM tree
    tree = parse(datadir+"/"+f)

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

            sentences_list.append(tokens_list)
            temp_labels.append(ddi_type)

labels_list.append(temp_labels)    # convertim llista de lables a llista de llistes (una) de labels

print('\n\n** Llista FINAL ' + name_model + '**\n{}'.format(sentences_list))
print('\n\n** Labels FINAL ' + name_model + '**\n{}'.format(labels_list))

#Save lists to file
with open(name_model + '_sentences_list.data', 'wb') as file:
    # store the data as binary data stream
    pickle.dump(sentences_list, file)
with open(name_model + '_labels_list.data', 'wb') as file:
    pickle.dump(labels_list, file)


######################################
### Word2Vec -- create Word Embeddings
######################################

# Word-Embeddings for the sentences
model_sentences = gensim.models.Word2Vec(sentences_list, size=300, window=5, min_count=1, workers=10)
print('*** Vocabulary for word sentences built')
model_sentences.save(name_model + '_sentences.model')
print('*** Model for word sentences saved')

'''
model_sentences.train(sentences_list, total_examples=len(sentences_list), epochs=10)
print('*** Model for word sentences trained')
'''

# Word-Embeddings for the labels
model_labels = gensim.models.Word2Vec(labels_list, size=300, window=5, min_count=1, workers=10)
print('*** Vocabulary for labels built')
model_labels.save(name_model + '_labels.model')
print('*** Model for labels saved')

