# **SINGLE OBJECT DETECTION USING INCEPTION V3**

**Coded using Keras with Tensorflow backend.**


**FILES**

  1. single-object-detection-inception.py


FILE : single-object-detection-inception.py

* Loaded **InceptionV3** model with imagenet weight
* Loaded image using load_img() of keras preprocessing and reshape to **299x299** ( min size for Inception V3 )
* Convert image to array using **img_to_array()**
* Expanded the dimension using numpy
* Preprocessed the image as required by Inception using **preprocess_input()** function of inceptionv3 in keras
* Made prediction with the model
* Decoded the prediction using **decode_predictions()** of inceptionv3 in keras
* Over the original image displayed the prediction and confidence using opencv's **putText()** function

