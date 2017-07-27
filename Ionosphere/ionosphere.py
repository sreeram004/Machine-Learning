from keras.models import Sequential
from keras.layers import Dense
import numpy
from sklearn import model_selection
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from keras.layers import Dropout

seed = 7
numpy.random.seed(seed)

dataset = numpy.loadtxt("ionosphere.csv", delimiter=",")

# split into input (X) and output (Y) variables
X = dataset[:, 0:34].astype(float)
Y = dataset[:, 34]


# Split X and Y arrays to X_train, X_validation, Y_train, Y_validation for training and validation
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size,
                                                                                random_state=seed)


def baseline_model():
    model = Sequential()
    model.add(Dense(40, input_dim=34, kernel_initializer='uniform', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

    model.compile(loss='binary_crossentropy' , optimizer='adam', metrics=['accuracy'])

    return model

# Fit the model
#model.fit(X_train, Y_train, epochs=1000, batch_size=10)

estimator = KerasClassifier(build_fn=baseline_model, epochs=50, batch_size=5, verbose=0)
estimator.fit(X, Y)

test = numpy.array([[1,	0,	1,	-0.18829,	0.93035,	-0.36156,	-0.10868,	-0.93597,	1,	-0.04549,	0.50874,	-0.67743,
                     0.34432,	-0.69707,	-0.51685,	-0.97515,	0.05499,	-0.62237,	0.33109,	-1,	-0.13151,	-0.453,
                     -0.18056,	-0.35734,	-0.20332,	-0.26569,	-0.20468,	-0.18401,	-0.1904,	-0.11593,	-0.16626,
                     -0.06288,	-0.13738,	-0.02447]])

prediction = estimator.predict(test)

print(prediction)

kFold = KFold(n_splits=10, shuffle=True, random_state=seed)

results = cross_val_score(estimator, X, Y, cv=kFold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))