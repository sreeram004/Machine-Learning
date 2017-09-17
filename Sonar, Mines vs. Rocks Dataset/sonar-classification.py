import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn import model_selection
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

seed = 7
numpy.random.seed(seed)

data = pandas.read_csv('sonar.csv', delimiter=',', header=None)

# print(data.groupby(60).size())

dataset = data.values

X = dataset[:, 0:60].astype(float)
Y = dataset[:, 60]

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split X and Y arrays to X_train, X_validation, Y_train, Y_validation for training and validation
validation_size = 0.33
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size,
                                                                                random_state=seed)

# print(X_validation.shape)
# print(Y_validation.shape)


seed = 7
scoring = 'accuracy'


def nn_model():

    model = Sequential()

    model.add(Dense(100, input_dim=60, kernel_initializer='normal', activation='relu'))

    model.add(Dense(36, kernel_initializer='normal', activation='relu'))
    model.add(Dense(26, kernel_initializer='normal', activation='relu'))
    model.add(Dense(10, kernel_initializer='normal', activation='relu'))

    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


testData = numpy.array([[0.0453,0.0523,0.0843,0.0689,0.1183,0.2583,0.2156,0.3481,0.3337,0.2872,0.4918,0.6552,0.6919,0.7797,0.7464,0.9444,1.0000,0.8874,0.8024,0.7818,0.5212,0.4052,0.3957,0.3914,0.3250,0.3200,0.3271,0.2767,0.4423,0.2028,0.3788,0.2947,0.1984,0.2341,0.1306,0.4182,0.3835,0.1057,0.1840,0.1970,0.1674,0.0583,0.1401,0.1628,0.0621,0.0203,0.0530,0.0742,0.0409,0.0061,0.0125,0.0084,0.0089,0.0048,0.0094,0.0191,0.0140,0.0049,0.0052,0.0044
                         ]])

testData = scaler.fit_transform(testData)

model = nn_model()

history = model.fit(X_train, Y_train, validation_data=(X_validation, Y_validation), batch_size=25, verbose=1, epochs=50, shuffle=True)


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

testData = scaler.transform(testData)

prediction = model.predict_classes(testData)
print(prediction)



