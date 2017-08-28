# BANK NOTE CLASSIFICATION
**DOMAIN** : BINARY CLASSIFICATION
**SOURCE** : UCI
**URL** : https://archive.ics.uci.edu/ml/datasets/banknote+authentication

**ACCURACY : 100% (epochs v/s accuracy graph attatched for verification)**

CODED IN PYTHON WITH KERAS IN PYCHARM IDE
BACKEND : Theano

**FILES**

1. bank-note.py
2. banknote.csv
3. banknote-classification.png

**FILE** : bank-note.py

* Loaded the dataset from csv file using pandas
* Observed the histogram and nature of the dataset
* Loaded the dataset
* Splitted the dataset in the ratio 0.2 for test and train
* Defined a simple nerual net as
     - **8->2**
* Final layer with softmax activation
* Compiled the model with adam optimizer
* fit the model
* Evaluated the model ( got 100% accuracy)
* Displayed the training plot (epochs v/s accuracy)
* Made predictions on the testData chosen from the dataset

**|| Split ratio was increased to .30 and accuracy changed to 99.2%**
**|| Split ratio was further increasd to .50 and accuracy changed to 96.8%**

