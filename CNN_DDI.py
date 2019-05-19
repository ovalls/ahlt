
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
import pickle
import numpy as np

import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D


# Load sentences and labels models for train, val and test
train_sentences_model = Word2Vec.load("train_sentences.model")
val_sentences_model = Word2Vec.load("val_sentences.model")
test_sentences_model = Word2Vec.load("test_sentences.model")

train_labels_model = Word2Vec.load("train_labels.model")
val_labels_model = Word2Vec.load("val_labels.model")
test_labels_model = Word2Vec.load("test_labels.model")


# Load list of lists of words and labels (classes) for each sentence, for train, val and test
with open('train_sentences_list.data', 'rb') as file:
    # read the data as binary data stream
    train_sentences_list = pickle.load(file)
with open('val_sentences_list.data', 'rb') as file:
    val_sentences_list = pickle.load(file)
with open('test_sentences_list.data', 'rb') as file:
    test_sentences_list = pickle.load(file)

with open('train_labels_list.data', 'rb') as file:
    train_labels_list = pickle.load(file)
with open('val_labels_list.data', 'rb') as file:
    val_labels_list = pickle.load(file)
with open('test_labels_list.data', 'rb') as file:
    test_labels_list = pickle.load(file)

# a veure si s'han obert ok -- perfekt!
print('val_sentences_list:\n{}'.format(val_sentences_list))

# Train sentences and labels for train, val and test
train_sentences_model.train(train_sentences_list, total_examples=len(train_sentences_list), epochs=10)
print('*** Model for train word sentences trained')
val_sentences_model.train(val_sentences_list, total_examples=len(val_sentences_list), epochs=10)
print('*** Model for val word sentences trained')
test_sentences_model.train(test_sentences_list, total_examples=len(test_sentences_list), epochs=10)
print('*** Model for test word sentences trained')

train_labels_model.train(train_labels_list, total_examples=len(train_labels_list), epochs=10)
print('*** Model for train labels trained')
val_labels_model.train(val_labels_list, total_examples=len(val_labels_list), epochs=10)
print('*** Model for val labels trained')
test_labels_model.train(test_labels_list, total_examples=len(test_labels_list), epochs=10)
print('*** Model for test labels trained')

'''
print('dose: {}'.format(val_sentences_model.wv['dose']))
print('DrugY: {}'.format(val_sentences_model.wv['DrugY']))
print('approximately: {}'.format(val_sentences_model.wv['approximately']))
print('coadministration: {}'.format(val_sentences_model.wv['coadministration']))

print('\nmodel sentences:\n{}'.format(val_sentences_model.wv))
print('\nraw vector arrays of sentences:\n{}'.format(val_sentences_model.wv.syn0))
print('len tot array: {}'.format(len(val_sentences_model.wv.syn0)))
print('len array[0]: {}'.format(len(val_sentences_model.wv.syn0[0])))

print('\nraw vector arrays of labels:\n{}'.format(val_labels_model.wv.syn0))
print('len tot array: {}'.format(len(val_labels_model.wv.syn0)))
print('len array[0]: {}'.format(len(val_labels_model.wv.syn0[0])))

#print('\nlist of words, index-ordered:\n{}'.format(model.wv.index2word))

print('\nmodel labels:\n{}'.format(val_labels_model.wv.syn0))
print('\nraw vector arrays of labels:\n{}'.format(val_labels_model.wv.syn0))
print(len(val_labels_model.wv.syn0))

print('advise: {}'.format(val_labels_model.wv['advise']))
print('considered: {}'.format(val_sentences_model.wv['considered']))

#print('mechanism: {}'.format(model_labels.wv['mechanism']))
#print('effect: {}'.format(model_labels.wv['effect']))
#print('no-interaction: {}'.format(model_labels.wv['no-interaction']))
'''


### Donats els vocabularis, crear una matriu 2D on cada row es una word (vector 300 dimensions)
# Posem una paraula a sota de l'altra, i una sentence darrera l'altra
# Ho fem per a train, val i test
# Primer hem de fer padding per a que totes les sentences ocupin el mateix número de words (181)

#print('max length sentence train: {}'.format(len(max(train_sentences_list,key=len))))
#print('max length sentence val: {}'.format(len(max(val_sentences_list,key=len))))
#print('max length sentence test: {}'.format(len(max(test_sentences_list,key=len))))

# Padding:
maxlen_train = len(max(train_sentences_list,key=len))
maxlen_val = len(max(val_sentences_list,key=len))
maxlen_test = len(max(test_sentences_list,key=len))
maxlen = max(maxlen_train, maxlen_val, maxlen_test)
print(maxlen)
maxlen = int(maxlen + 0.1*maxlen)
print(maxlen)

train_vector_sentences = []
train_vector_labels = []
val_vector_sentences = []
val_vector_labels = []
test_vector_sentences = []
test_vector_labels = []

# llista de llistes de word embeddings de les SENTENCES de TRAIN
for s in train_sentences_list:
    sentence = []
    for w in s:
        sentence.append(train_sentences_model.wv[w])
    train_vector_sentences.append(sentence)

x_train= []
# zero padding to maxlen (max length of words in a sentence = 181)
for s in range(0, len(train_vector_sentences)):
    words_left = maxlen - len(train_vector_sentences[s])
    for i in range(0,words_left):
        train_vector_sentences[s].append(np.zeros(300))
    for ws in train_vector_sentences[s]:    # Entrada CNN: una sentence darrera de l'altra (2D matrix)
        x_train.append(ws)

# llista de llistes de word embeddings de les SENTENCES de VAL
for s in val_sentences_list:
    sentence = []
    for w in s:
        #word = []
        #word.append(val_sentences_model.wv[w])
        sentence.append(val_sentences_model.wv[w])
    val_vector_sentences.append(sentence)               # word embeddings de cada sentence
#print('len val_vector_sentences: {}'.format(len(val_vector_sentences)))
#print(val_vector_sentences[0])
#print('len val_vector_sentences[0]: {}'.format(len(val_vector_sentences[0])))
#print('len val_vector_sentences[1]: {}'.format(len(val_vector_sentences[1])))

x_val= []
# zero padding to maxlen (max length of words in a sentence = 181)
for s in range(0, len(val_vector_sentences)):
    words_left = maxlen - len(val_vector_sentences[s])
    #print('words left: {}'.format(words_left))
    for i in range(0,words_left):
        val_vector_sentences[s].append(np.zeros(300))
    #print('len: {}'.format(len(val_vector_sentences[s])))
    for ws in val_vector_sentences[s]:
        x_val.append(ws)

#print('val_vector_sentences[0]: {}'.format(val_vector_sentences[0]))

# llista de llistes de word embeddings de les SENTENCES de TEST
for s in test_sentences_list:
    sentence = []
    for w in s:
        sentence.append(test_sentences_model.wv[w])
    test_vector_sentences.append(sentence)

x_test = []
# zero padding to maxlen (max length of words in a sentence = 181)
for s in range(0, len(test_vector_sentences)):
    words_left = maxlen - len(test_vector_sentences[s])
    for i in range(0,words_left):
        test_vector_sentences[s].append(np.zeros(300))
    for ws in test_vector_sentences[s]:
        x_test.append(ws)


# llista de word embeddings de les LABELS de TRAIN
for l in train_labels_list:
    sentence = []
    for i in l:
        sentence.append(train_labels_model.wv[i])
    train_vector_labels.append(sentence)

y_train = []
for sl in train_vector_labels[0]:   #només té un element que és una llista de la classe de cada sentence
    y_train.append(sl)

# llista de word embeddings de les LABELS de VAL
for l in val_labels_list:
    sentence = []
    for i in l:
        sentence.append(val_labels_model.wv[i])
    val_vector_labels.append(sentence)

y_val = []
for sl in val_vector_labels[0]:
    y_val.append(sl)

#print('\nval_vector_labels: {}'.format(val_vector_labels))

# llista de word embeddings de les LABELS de TEST
for l in test_labels_list:
    sentence = []
    for i in l:
        sentence.append(test_labels_model.wv[i])
    test_vector_labels.append(sentence)

y_test = []
for sl in test_vector_labels[0]:
    y_test.append(sl)

print('\nLen Sentences Train: {}'.format(len(train_vector_sentences)))
print('Len Labels Train: {}'.format(len(train_vector_labels)))
print('len Sentences Val: {}'.format(len(val_vector_sentences)))
print('Len Labels Val: {}'.format(len(val_vector_labels)))
print('Len Sentences Test: {}'.format(len(test_vector_sentences)))
print('Len Labels Test: {}'.format(len(test_vector_labels)))
print('len x_train: {}'.format(len(x_train)))
print('len x_val: {}'.format(len(x_val)))
print('len x_test: {}'.format(len(x_test)))
print('len y_train: {}'.format(len(y_train)))
print('len y_val: {}'.format(len(y_val)))
print('len y_test: {}'.format(len(y_test)))


################################
### Convolutional Neural Network
################################

print('\n*** Configuring CNN')
# vocabulary, batch, epochs
max_words, batch_size, epochs = 1000, 64, 2
# dimension of dense embedding
#embedding_dims = 50
embedding_dims = 300                            # crec que ha de ser com l'he definit al model ¿?¿?¿?
#max length of sentence --> computed before = 181
filters, kernel_size, hidden_dims = 250, 5, 150

num_classes = len(val_sentences_model.wv.syn0[0])   # We have 5 different classes

## Vectorize the output sentence type classifcations to Keras readable format
#train_vector_labels = keras.utils.to_categorical(train_vector_labels, num_classes)
#val_vector_labels = keras.utils.to_categorical(val_vector_labels, num_classes)
#test_vector_labels = keras.utils.to_categorical(test_vector_labels, num_classes)

# Pad the input vectors to ensure a consistent length
# Li passo una llista de llistes (de números)
# Jo tinc llista de sentences; cada sentences és llista de paraules; cada paraula és array,
# per tant li he de passar cada sentence de la llista de sentences
#for s in range(0,len(train_vector_sentences)):
 #   train_vector_sentences[s] = sequence.pad_sequences(train_vector_sentences[s], maxlen=maxlen)


model = Sequential()

# Created Embedding (Input) Layer (max_words) --> Convolutional Layer
# Turns positive integers (indexes) into dense vectors of fixed size. eg. [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]
# This layer can only be used as the first layer in a model.
# Input shape: 2D tensor with shape: (batch_size, sequence_length).
# Output shape: 3D tensor with shape: (batch_size, sequence_length, output_dim).

model.add(Embedding(max_words, embedding_dims, input_length=maxlen))
model.add(Dropout(0.2)) # masks various input values

# Create the convolutional layer
model.add(Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1))

# Create the pooling layer
model.add(GlobalMaxPooling1D())

# Create the fully connected layer
model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('relu'))

#model.add(Flatten())

# Create the output layer (num_classes)
#model.add(Dense(num_classes))
#model.add(Activation('softmax'))
model.add(Dense(num_classes, activation='softmax'))

# Add optimization method, loss function and optimization value
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# "Fit the model" (train model), using training data
#model.fit(train_vector_sentences, train_vector_labels, batch_size=batch_size,
 #         epochs=epochs, validation_data=(test_vector_sentences, test_vector_labels))
model.fit(x_train, y_train, batch_size=batch_size,
          epochs=epochs, validation_data=(x_test, y_test))

# ValueError: Error when checking input:
# expected embedding_1_input to have 2 dimensions, but got array with shape (22522, 500, 300)

# Evaluate the trained model, using the test data (20% of the dataset)
#score = model.evaluate(test_vector_sentences, test_vector_labels, batch_size=batch_size)
score = model.evaluate(x_test, y_test, batch_size=batch_size)

print('Score: {}'.format(score))
