import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import preprocessing
from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import os

print(os.getcwd())

# Folder Path
path = "/root/autodl-tmp/data/aclImdb/train/neg"
# Change the directory
os.chdir(path)

train_neg = []

# Read text File
def read_text_file(file_path):
    with open(file_path, 'r') as f:
        train_neg.append(f.read())
  

# iterate through all file
for file in os.listdir():
    # Check whether file is in text format or not
    if file.endswith(".txt"):
        file_path = f"{path}/{file}"
  
        # call read text file function
        read_text_file(file_path)

path = "/root/autodl-tmp/data/aclImdb/train/pos"
# Change the directory
os.chdir(path)

train_pos = []

# Read text File
def read_text_file(file_path):
    with open(file_path, 'r') as f:
        train_pos.append(f.read())
  

# iterate through all file
for file in os.listdir():
    # Check whether file is in text format or not
    if file.endswith(".txt"):
        file_path = f"{path}/{file}"
  
        # call read text file function
        read_text_file(file_path)

print(len(train_pos)) # number of positive comments
print(len(train_neg)) # number of negative comments

texts_train = train_neg + train_pos
y_train_neg = [0]*len(train_neg)
y_train_pos = [1]*len(train_pos)
y_train = np.array(y_train_neg + y_train_pos)

print(texts_train[12499]) # negative
print(texts_train[12500]) # positive

vocabulary = 10000
tokenizer = Tokenizer(num_words=vocabulary)
tokenizer.fit_on_texts(texts_train) # build dictionary

word_index = tokenizer.word_index # word dictionary
sequence_train = tokenizer.texts_to_sequences(texts_train) # text to sequence

# align sequences
word_num = 20
X_train = preprocessing.sequence.pad_sequences(sequence_train, maxlen=word_num) # Pre-padding or removing values from the beginning of the sequence is the default.

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

embedding_dim = 8
model = Sequential()
model.add(Embedding(vocabulary, embedding_dim, input_length=word_num))
model.add(Flatten())
model.add(Dense(1,activation='sigmoid'))

model.summary()

epochs = 50
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),loss='binary_crossentropy',metrics=['acc'])

history = model.fit(X_train,y_train, epochs=epochs, batch_size=32, validation_data=(X_valid,y_valid))
comments = [texts_train[610]]
print(comments) # should be negative
sequence_prediction = tokenizer.texts_to_sequences(comments) # input should be list of sequences
xhat = preprocessing.sequence.pad_sequences(sequence_prediction, maxlen=word_num)
model.predict(xhat)