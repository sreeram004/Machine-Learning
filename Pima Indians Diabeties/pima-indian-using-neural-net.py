from keras.models import Sequential
from keras.layers import Dense
import numpy
from sklearn import model_selection
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler, Imputer
import pandas
import missingno as msn


# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load pima indians dataset
data = pandas.read_csv("pima-indians-diabetes.csv", delimiter=",", header=None)



# print(data.describe())
# print((data[[1,2,3,4,5]] == 0).sum())

#data[[1,2,3,4,5]] = data[[1,2,3,4,5]].replace(0, numpy.NaN)
#msn.bar(data)
# print(data.isnull().sum())

# print(data.head(20))

dataset = data.values

imputer = Imputer()
dataset = imputer.fit_transform(dataset)

# print(numpy.isnan(dataset).sum())

# split into input (X) and output (Y) variables
X = dataset[:, 0:8].astype(float)
Y = dataset[:, 8]

# print(numpy.isnan(X).sum())


scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.20, random_state=seed)

# create model
model = Sequential()
model.add(Dense(12, input_dim=8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# Fit the model
model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=1, validation_data=(X_test, y_test))

scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

test = numpy.array([[1, 111, 62, 13, 182, 24, 0.138, 23]])
prediction = model.predict_classes(test)


