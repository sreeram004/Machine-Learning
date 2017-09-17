#Sonar, Mines vs. Rocks Dataset

**SOURCE** : UCI

**URL** : http://archive.ics.uci.edu/ml/datasets/connectionist+bench+(sonar,+mines+vs.+rocks)

**DOMAIN** : **BINARY CLASSIFICATION**

**DATA ITEMS : 208**

**FILES**

1.  sonar-classification.py
2. sonar.csv
3. sonar-accuracy-plot.png

**FILE** : sonar-classification.py

 * Fixed random seed
 * Loaded dataset from csv file using pandas
 * Observed the histogram and checked for missing values using numpy
 * **Standard Scaled** dataset
 * Split dataset in to train and validation sets (0.33 ratio)
 * Defined a neural net with following neurons per layer
      - **100 -> 36 -> 26 ->  10 -> 1**
 * Final layer activated with **sigmoid** and others with **Rectifier** (relu)
 * Compiled model with **binary_crossentropy** loss with **adam** optimizer
 * Fitted the model with training set and trained for **50** epochs
 * Evaluated the model with validation data (**91.30 %**)
 * Plotted accuracy plot using training history
 * Loaded a random data and **Standard Scaled** it
 * Made predictions with that data
 







