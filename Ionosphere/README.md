**IONOSPHERE DATASET**

**SOURCE** : UCI
**DATA ITEMS** : 351

**ACCURACY : 94.29%**

**STEPS**

 * Dataset is divided in to training and validation sets
 * **A neural net is build as 40->8->1 with sigmoid activation function for last layer**
 * Dropouts was added for layers
 * KerasClassifier wrapper is used to wrap the model
 * model was fitted on the dataset
 * **kFold validation with k = 10 was done and the mean and standard deviation of the result was printed**
 * External data is added in the testData numpy array 
 * Prediction is printed
