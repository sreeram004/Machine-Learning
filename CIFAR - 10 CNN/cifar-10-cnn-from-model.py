from keras.models import load_model
import cv2
import numpy as np


model = load_model('cifar-10-model.h5')

test_image = cv2.imread('horse.jpg')
test_image = cv2.resize(test_image, (32, 32))
test_image = np.array(test_image)

test_image = test_image.astype('float32')
test_image /= 255

print(test_image.shape)

test_image = np.rollaxis(test_image, 2, 0)
test_image = np.expand_dims(test_image, axis=0)
print(test_image.shape)

prediction = model.predict_classes(test_image)
print(prediction)
