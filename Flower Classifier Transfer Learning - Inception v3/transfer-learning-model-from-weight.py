import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn import model_selection
from keras import backend as K
from keras.constraints import maxnorm
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from keras.callbacks import ModelCheckpoint
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Input
from keras.models import Model
import cv2

K.set_image_dim_ordering('tf')

seed = 7
np.random.seed(seed)

PATH = os.getcwd()
# Define data path
data_path = PATH + '\lower_photos'
data_path = data_path + '\\'
data_dir_list = os.listdir(data_path)

img_rows = 140
img_cols = 140
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
print(img_data.shape)

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
dummy_y = np_utils.to_categorical(encoded_Y, num_classes)

seed = 7


x, y = shuffle(img_data, dummy_y, random_state=seed)

X_train, X_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2, random_state=seed)

input_shape = img_data[0].shape


print(X_train.shape)
print(X_test.shape)

print(y_train.shape)
print(y_test.shape)


model = InceptionV3(include_top=False, classes=5, input_shape=input_shape, weights='imagenet', pooling='avg')

predictions = Dense(5, activation='softmax', name='predictions')(model.output)
model = Model(inputs=model.input, outputs=predictions)

model.load_weights('weights-improvement-transfer-learning-tf-05-0.83.h5')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# print(model.summary())


scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


model.save('transfer-learning-model-flower.h5', overwrite=True)


test_image = cv2.imread('rose01.jpg')
# test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
test_image = cv2.resize(test_image, (img_rows, img_cols))
test_image = np.array(test_image)
test_image = test_image.astype('float32')
test_image /= 255
print(test_image.shape)

if num_channel == 1:
    if K.image_dim_ordering() == 'th':
        test_image = np.expand_dims(test_image, axis=0)
        test_image = np.expand_dims(test_image, axis=0)
        print(test_image.shape)
    else:
        test_image = np.expand_dims(test_image, axis=3)
        test_image = np.expand_dims(test_image, axis=0)
        print(test_image.shape)

else:
    if K.image_dim_ordering() == 'th':
        test_image = np.rollaxis(test_image, 2, 0)
        test_image = np.expand_dims(test_image, axis=0)
        print(test_image.shape)
    else:
        test_image = np.expand_dims(test_image, axis=0)
        print(test_image.shape)

# Predicting the test image

prediction = model.predict(test_image)

print(np.argmax(prediction, axis=1))

