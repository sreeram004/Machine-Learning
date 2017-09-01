# FLOWER CLASSIFFICATION - TRANSER LEARNING

**DATASET** : FROM TENSORFLOW WEBISTE. 

**URL** : http://download.tensorflow.org/example_images/flower_photos.tgz

**CLASSES** : 5 {Daisy, Dandelion, Rose, Sunflower, Tulips}

**IMAGES** : 2871 {Total}

CODED IN PYTHON WITH KERAS WITH PYCHARM IDE

**NB : WEIGHT FILE AND SAVED MODEL NOT ATTATCHED DUE TO LARGE SIZE (200+ AND 80+ MB RESPECTIVELY)**

**BACKEND : TENSORFLOW**

**ACCURACY : 83% (7 EPOCHS)**


**FILES**
1. transfer-learning-flowers.py
2. transfer-learning-model-from-weight.py
3. transfer-learning-model-make-prediction.py
4. Test Images Folder - contains images used for make prediction with saved model for testing purpose

**FILE** : transfer-learning-flowers.py
**OUTPUT** : trained model's weight 

* Randomness fixed with seed
* Specified the rows and columns as **140** (to use **140x140** image - minimum for **inception** **v3**)
* Color channels as 3 (For **RGB**)
* Loaded the downloaded dataset situated at the home folder after resizing to **140x140**
* Changed the image dimension ordering for the tensorflow backend
* Declared the labels array with the corresponding flower value ( Daisy - 0, Dandelion - 1, so on)
* One hot encoded labels and stored to dummy_y
* Shuffeled the dataset with randomness provided by seed
* Split the dataset to test and train
* Loaded the **inception** **v3** model with imagenet weight for 5 classes and avg pooling
* **Changed the last layer to predict 5 classed instead of 1000 usind Dense layer of 5 nodes**
* Compiled using adam optimizer
* Defined checkpoints to save the best weight
* Displayed the model summary
* fit the model with training sets
* plotted the epochs v/s accuracy graph
* Evaluated the model printing the accuracy
* Loaded external image to test
* Made prediction and displayed it after rounding off usifng argmax() of numpy


**FILE** : transfer-learning-model-from-weight.py

* Randomness fixed with seed
* Loaded the dataset
* Defined the model as before
* Loaded the saved weight
* Compiled the model
* Saved the model

**FILE** : transfer-learning-model-make-prediction.py

* Loaded the saved model
* Loaded the image to make prediction
* Applyed needed image dimension ordering
* Made precdiction and displayed


**TUNINGS TO PERFORM**

* Accuracy tend to imporove with increasing the dimension
* More epochs could give a better model
* Freezing some layers of the model could increase the accuracy
