from keras.models import Sequential
from keras.layers import LSTM, Embedding, Dense, Flatten
from keras import optimizers

epochs = 3
vocabulary = 10000 # unique words in the dictionary
embedding_dim = 32 # embedding 向量维度
word_num = 500     # sequence length
state_dim = 32     # RNN 状态向量维度

# Stacked LSTM
model = Sequential()
model.add(Embedding(vocabulary, embedding_dim, input_length=word_num))
model.add(LSTM(state_dim, return_sequences=True, dropout=0.2))  # 只返回最后一个状态
model.add(LSTM(state_dim, return_sequences=True, dropout=0.2))
model.add(LSTM(state_dim, return_sequences=False, dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.summary()

# model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['ass'])
# history = model.fit(x_train, y_train, epochs=epochs,batch_size=32, validation_data=(x_valid,y_valid))
