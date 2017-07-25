# Load libraries
import pandas
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import numpy

# Load dataset from the given url
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv('iris.csv', names=names)

'''
# print shape of dataset
print(dataset.shape)

# first 20 items of dataset
print(dataset.head(20))

# descriptions of dataset
print(dataset.describe())

# class distribution of dataset
print(dataset.groupby('class').size())

# box and whisker plots of the dataset
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()



# histograms of dataset
dataset.hist()
plt.show()

'''

# Split-out validation dataset in to 2 numpy arrays X and Y
array = dataset.values
X = array[:, 0:4]
Y = array[:, 4]

# Split X and Y arrays to X_train, X_validation, Y_train, Y_validation for training and validation
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size,
                                                                                random_state=seed)

'''
# Test options and evaluation metric
seed = 7
scoring = 'accuracy'

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

'''


# Making predictions on validation dataset using KNN and printing score and matrix

'''knn = KNeighborsClassifier()
knn.fit(X, Y)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))'''

# create a test array to check
testData = numpy.array([[4.6, 3.2, 1.4, 0.2]])
# using Support Vector Machine as it gave best accuracy in my Laptop
svd = SVC()
svd.fit(X, Y)  # final model - trained on full dataset
predictions = svd.predict(testData)  # make prediction on test data we defined
print(predictions)  # print the prediction

#print(accuracy_score(Y_validation, predictions))  # print accuracy score
#print(confusion_matrix(Y_validation, predictions))  # print the confusion matrix
#print(classification_report(Y_validation, predictions))  # print the report