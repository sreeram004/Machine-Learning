# YOUTUBE SPAM CLASSIFICATION

**SOURCE** : UCI

**URL** : https://archive.ics.uci.edu/ml/datasets/YouTube+Spam+Collection

**TYPE : BINARY CLASSIFICATION**

**BACKEND** : THEANO

**ACCURACY : 93.50%**

**FILES**
 
 1. youtube-spam-classification-cnn_lstm.py
 2. Youtube01-Psy.csv
 3. Youtube02-KatyPerry.csv
 4. Youtube03-LMFAO.csv
 5. Youtube04-Eminem.csv
 6. Youtube05-Shakira.csv

**FILE** : youtube-spam-classification-cnn_lstm.py

 * Fixed the random seed 
 * Loaded datasets as different dataframes
 * Concatinate all dataframes into one
 * Dropped unnecessary columns
 * Seperated dataset to X and Y
 * Create a tokenizer object
 * Fit tokenizer on X and converted to sequence
 * Split the dataset to train and test sets
 * Pad train and test sets to maxlength 120
 * Create a model as follows
 	
    * Embedding layer with vocabulary 5000, 32 as vector length and max length as 120
    * Conv1D layer with 32 filters, kernal size 3 and activation relu
    * MaxPooling1D layer with 2 as pool size
    * LSTM layer with 100 units with dropouts 0.2
    * Dense layer of 1 node with sigmoid activation
 * Compiled the model with binary_crossentropy loss
 * Fitted the model on training sets and validated on validation sets for 3 epochs and batch size 5
 * Evaluated the model on validation set
 * Displayed the accuracy (93.50%)
 * Plotted the training history
 * Assigned test sample as list item
 * Fitted tokenizer on test and converted to sequence
 * Pad the sequence to have 120 as length
 * Make prediction
 * If prediction > 0.5 then class is 1 else class is 0
 