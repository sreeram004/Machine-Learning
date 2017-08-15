import numpy
import pandas
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection

names = ['Variance', 'Skewness', 'Kurtosis', 'Entropy', 'Class']
dataFrame = pandas.read_csv('banknote.csv', header=None, delimiter=',', names=names)
# print(dataFrame.head(10))

# print(dataFrame.groupby('Class').size())

dataset = dataFrame.values

# dataFrame.hist()
# plt.show()



X = dataset[:, 0:4]
Y = dataset[:, 4]

# print(Y)

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

validation_size = 0.2
seed = 7

X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, dummy_y, test_size=validation_size,
                                                                                random_state=seed)


def nnmodel():

    model = Sequential()

    model.add(Dense(8, input_dim=4, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = nnmodel()

print(model.summary())

history = model.fit(X_train, Y_train, batch_size=20, verbose=1, epochs=50, validation_data=(X_validation, Y_validation))


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

testData = numpy.array([[-2.9821,4.1986,-0.5898,-3.9642]])
prediction = model.predict_classes(testData)

print(prediction)
print(encoder.inverse_transform(prediction))
