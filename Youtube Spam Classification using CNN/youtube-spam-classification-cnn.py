from pandas import DataFrame
from pandas import read_csv, concat
import numpy
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

dataframe1 = read_csv('Youtube05-Shakira.csv', delimiter=',')
dataframe2 = read_csv('Youtube04-Eminem.csv', delimiter=',')
dataframe3 = read_csv('Youtube03-LMFAO.csv', delimiter=',')
dataframe4 = read_csv('Youtube02-KatyPerry.csv', delimiter=',')
dataframe5 = read_csv('Youtube01-Psy.csv', delimiter=',')

frames = [dataframe1, dataframe2, dataframe3, dataframe4, dataframe5]

dataframe = concat(frames)

dataframe.drop('COMMENT_ID', axis=1, inplace=True)
dataframe.drop('AUTHOR', axis=1, inplace=True)
dataframe.drop('DATE', axis=1, inplace=True)

# print(dataframe.head(5))
# print(numpy.unique(dataframe['CLASS']))

dataset = dataframe.values

X = dataset[:, 0]
Y = dataset[:, 1]

# Summarize number of words
print("Number of words: ")
print(len(numpy.unique(numpy.hstack(X))))


tokenizer = Tokenizer()

tokenizer.fit_on_texts(X)
X = tokenizer.texts_to_sequences(X)

validation_size = 0.33
seed = 7

X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size,
                                                                                random_state=seed)

X_train = sequence.pad_sequences(X_train, maxlen=120)
X_validation = sequence.pad_sequences(X_validation, maxlen=120)

# print(X_train[0])

total_words = 5000
max_length = 120
embedding_vector_length = 32

# create the model
model = Sequential()
model.add(Embedding(total_words, embedding_vector_length, input_length=max_length))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())

model.add(Dense(250, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# Fit the model
history = model.fit(X_train, Y_train, validation_data=(X_validation, Y_validation), epochs=3, batch_size=5, verbose=1)
# Final evaluation of the model
scores = model.evaluate(X_validation, Y_validation, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

test = ['Check out this video on YouTube:']
test1 = test

tokenizer.fit_on_texts(test)
test = tokenizer.texts_to_sequences(test)

test = sequence.pad_sequences(test, maxlen=120)

print(numpy.greater(model.predict(test), 0.5) * 1.0)
