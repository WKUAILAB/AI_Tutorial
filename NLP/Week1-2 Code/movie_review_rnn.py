import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import preprocessing
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, Embedding
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import os

tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

print(os.getcwd())

# Folder Path
path = "/root/autodl-tmp/Data/aclImdb/train/neg"
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

path = "/root/autodl-tmp/Data/aclImdb/train/pos"
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

vocabulary = 10000
tokenizer = Tokenizer(num_words=vocabulary)
tokenizer.fit_on_texts(texts_train) # build dictionary

word_index = tokenizer.word_index # word dictionary
sequence_train = tokenizer.texts_to_sequences(texts_train) # text to sequence

# align sequences
word_num = 500 # different with logistic regression
X_train = preprocessing.sequence.pad_sequences(sequence_train, maxlen=word_num) 


X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

embedding_dim = 32
state_dim = 32

model = Sequential()
model.add(Embedding(vocabulary, embedding_dim, input_length=word_num))
model.add(SimpleRNN(state_dim, return_sequences=False)) # only return last state
model.add(Dense(1, activation='sigmoid'))

model.summary()

epochs = 3 # avoid overfitting
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['acc'])
history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_valid, y_valid))