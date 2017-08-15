import os,cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn import model_selection
from keras import backend as K
K.set_image_dim_ordering('th')
from keras.constraints import maxnorm
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from keras.callbacks import ModelCheckpoint

PATH = os.getcwd()
# Define data path
data_path = PATH + '\lower_photos'
data_path = data_path + '\\'
data_dir_list = os.listdir(data_path)

img_rows = 32
img_cols = 32
num_channel = 3

# Define the number of classes
num_classes = 5

img_data_list=[]


# loading from all folders

for dataset in data_dir_list:
    img_list = os.listdir(data_path+dataset)
    # print(data_path+dataset)
    print('Loaded the images of dataset-' + '{}\n'.format(dataset))

    for img in img_list:
        # print(data_path+dataset+'\\' +img)
        input_img = cv2.imread(data_path+dataset+'\\'+img)
        # input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        input_img_resize = cv2.resize(input_img, (img_rows, img_cols))
        img_data_list.append(input_img_resize)

img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data /= 255
print (img_data.shape)

if num_channel == 1: # for gray scale
    if K.image_dim_ordering() == 'th': # for theano
        img_data = np.expand_dims(img_data, axis=1)
        print(img_data.shape)
    else: # for tensorflow
        img_data = np.expand_dims(img_data, axis=4)
        print(img_data.shape)

else: # for rgb
    if K.image_dim_ordering() == 'th': # for theano
        img_data = np.rollaxis(img_data, 3, 1)
        print(img_data.shape)

num_classes = 5

num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples,), dtype='int64')

labels[0:633] = 0
labels[633:1531] = 1
labels[1531:2172] = 2
labels[2172:2871] = 3
labels[2871:] = 4

names = ['daisy', 'dandelion', 'roses', 'sunflower', 'tulips']

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(labels)
encoded_Y = encoder.transform(labels)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y, num_classes)

seed = 7

#Shuffle the dataset
x, y = shuffle(img_data, dummy_y, random_state=seed)
# Split the dataset
X_train, X_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2, random_state=seed)

input_shape = img_data[0].shape

model = Sequential()

model.add(Convolution2D(64, (3, 3), padding='same', input_shape=input_shape, activation='relu'))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Convolution2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(50, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

lr = 0.01
epochs = 10
decay = lr/epochs


sgd = SGD(lr=lr, decay=decay, momentum=0.9, nesterov=False)
# model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])
model.load_weights("weights-improvement-11-0.72.h5")
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


# save the model for future use
model.save('cnnmodelextplotadamfromweight72.h5', overwrite=True)
print("Saved model to disk")
