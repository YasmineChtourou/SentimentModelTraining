import os
import sys
import re
import pickle
import numpy as np
import pandas as pd


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split


import preprocessing

DIR_GLOVE = os.path.abspath('../Glove/')
DIR_DATA = os.path.abspath('../Dataset/')
MAX_SEQUENCE_LENGTH = 100
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 300
TEST_SPLIT = 0.3
VALIDATION_SPLIT = 0.3
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
label_dict = {}
classes=[]


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string) 
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

# Load the glove file
def gloveVec(filename):
    embeddings = {}
    f = open(os.path.join(DIR_GLOVE, filename), encoding='utf-8')
    i = 0
    for line in f:
        values = line.split()
        word = values[0]
        try:
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings[word] = coefs            
        except ValueError:
            i += 1
    f.close()
    return embeddings

# load the dataset 
def loadData(filename):
    df = pd.read_csv(DIR_DATA + filename,delimiter=';')
    selected = ['Label', 'Text']
    non_selected = list(set(df.columns) - set(selected))
    # delete non_selected columns
    df = df.drop(non_selected, axis=1)
    df = df.dropna(axis=0, how='any', subset=selected)
    classes = sorted(list(set(df[selected[0]].tolist())))
    # classes = ['negative', 'neutre', 'positive']
    for i in range(len(classes)):
        label_dict[classes[i]] = i
        # label_dict = {'negative': 0, 'neutre': 1, 'positive': 2}
    sentences = df[selected[1]].apply(lambda x: clean_str(x)).tolist()
    labels = df[selected[0]].apply(lambda y: label_dict[y]).tolist()
    labels = to_categorical(np.asarray(labels))
    # to_categorical: Converts a class vector (integers) to binary class matrix
    return sentences,labels


def createVocabAndData(sentences):
    tokenizer.fit_on_texts(sentences)
    sequences = tokenizer.texts_to_sequences(sentences)
    vocab = tokenizer.word_index
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    return vocab,data

def createEmbeddingMatrix(word_index,embeddings_index):
    nb_words = min(MAX_NB_WORDS, len(word_index))
    embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i > MAX_NB_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def cnnModel(embedding_matrix,epoch):
    model = Sequential() # configure the model for training
    n, embedding_dims = embedding_matrix.shape
    
    model.add(Embedding(n, embedding_dims, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False))
     model.add(Conv1D(32,3, padding='valid', strides=1))
    model.add(GlobalMaxPooling1D())
    model.add(Dropout(0.6))
    model.add(Dense(3))
    model.add(Activation('softmax'))
    # add layers

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    
    history = model.fit(X_train, y_train, validation_split=VALIDATION_SPLIT, epochs=epoch, batch_size=64,callbacks=[EarlyStopping(patience=3)])
    # list all data in history
    plt.plot(history.history['acc'],color='blue')
    plt.plot(history.history['val_acc'],color='red')
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['Training Accuracy', 'Validation Accuracy'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'],color='blue')
    plt.plot(history.history['val_loss'],color='red')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Training Loss', 'Validation Loss'], loc='upper right')
    plt.show()

    scores= model.evaluate(X_test, y_test, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    return model

# load words from the file glove
embeddings = gloveVec('glove.840B.300d.txt')

sentences, labels = loadData('data.csv')

for i in range(len(sentences)):
    sentences[i] = preprocessing.transform_text(sentences[i])
vocab, data = createVocabAndData(sentences)

embedding_mat = createEmbeddingMatrix(vocab,embeddings)
pickle.dump([data, labels, embedding_mat], open('embedding_matrix.pkl', 'wb'))
print ("Data created")
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=TEST_SPLIT, random_state=42)

model = cnnModel(embedding_mat,50)
