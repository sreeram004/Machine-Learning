# Pima Indians Diabetes Data Set 

**SOURCE** : UCI

**URL** : https://archive.ics.uci.edu/ml/datasets/pima+indians+diabetes

**TYPE** : BINARY CLASSIFICATION

**MISSING VALUES** : YES

**ACCURACY OBTAINED : 81.17%**

**FILES**

1. pima-indians-diabetes.csv
2. pima-indian-using-neural-net.py

**FILE** : pima-indians-diabetes.csv

* Comma seperated dataset
* Contains missing values

**FILE** : pima-indian-using-neural-net.py

* Fixed random seed as 7 
* Read data using pandas
* **Described data and found missing values**
* **Found the number of missing values**
* **Replaced missing values with Nan and plotted it with missingno library**
* **Using Imputer from sklearn transformed the dataset to handle missing values problem**
* **StandardScaled the X**
* Splitted dataset to test and train sets on ratio 0.2
* **Defined a net as follows**
   * **layer 1 - 12 nodes - input layer - input dimension : 8 with uniform kernal initialization and relu activation**
   * **layer 2 - 8 nodes - hidden layer 1 - with uniform kernal initialization and relu activation**
   * **layer 3 - 1 node - output layer - with uniform kernal initialization and sigmoid activation**
* Compiled the model with binary_crossentropy loss and adam optimizer
* Fitted the model on training set and ran for **100 epochs**
* Evaluted the model and displayed accuracy (**81.17%**)
* Took a random data from the dataset and set it as a numpy array
* Made prediction based on that test array

