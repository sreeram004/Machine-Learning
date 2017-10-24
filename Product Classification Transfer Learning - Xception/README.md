**SOLUTION PREPARED FOR DEEP LEARNING CHALLENGE HOSTED BY HACKER EARTH**

**URL** : https://www.hackerearth.com/challenge/competitive/deep-learning-challenge-1/problems/

**DATASET** : https://he-s3.s3.amazonaws.com/media/hackathon/deep-learning-challenge-1/identify-the-objects/a0409a00-8-dataset_dp.zip

**FILE** : xception_tf_pdt.ipynb

* Loaded the dataset
* Loaded the train and test sets
* Loaded the Image data as **300x300** image
* Normalized data and Made the labels
* Loaded the **Xception** model on **imagenet** weight
* Added a **GlobalAveragePooling2D()** and **Dense** layer to the model
* Compiled model with **categorical_crossentropy** loss and **Adam** optimizer with **1e-4** learning rate
* Defined **ImageDataGenerator** with augementations like roatation, flips etc.
* Trained the model with **DataGenerator.flow()**
* Trained for 6 epochs and recorded the prediction to get the best result **(89.954% - 23rd Rank)**

