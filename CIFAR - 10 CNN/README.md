# **CIFAR - 10**

**Training Sample** : 50000

**Evaluation Sample** : 10000

**Coded using Keras with Theano backend.**

**Accuracy : 79.58%**

**FILES**

  1. cifar-10-cnn,py
  2. cifar-10-cnn-from-model.py
  3. cifar-10-model.h5

FILE : cifar-10-cnn.py

 * Loaded Dataset from Keras Datasets
 * Fixed random seed as 7
* Normalized inputs to range 0 - 1
* Enocoded output sets (y_train and y_test)
* **Defined cnn model with 6 Conv2D layers of 32, 32, 64, 64, 128, 128 filters and with kernal_size 3x3 MaxPooling2D layer of 2,2 with 0.2 Dropout**
  **Flattened and passed to the Dense layers 1024->512->10**
* Compiled with categorical_crossentropy loss and adam optimizer
* Fitted training set validating on the test set for 20epochs
* Evaluated the model to find the accuracy -  **Accuracy of 79.58% was obtained**
* Saved the model to disk for future use

  
FILE : cifar-10-cnn-from-model.py

 * Loaded the model
 * Loaded image using opencv
 * Resized and converted to numpy array
 * Changed type to float32
 * Normalized image
 * Rolled the axis to match image ordering of backend
 * Expanded dimensions for making prediction
 * Made prediction and displayed the result


