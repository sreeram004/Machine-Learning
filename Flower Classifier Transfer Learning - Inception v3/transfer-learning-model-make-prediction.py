# transfer learned model from inceptionv3 - 83% accuracy - 7 epochs - 140x140 images - tf backend and image ordering

import numpy as np
from keras.models import load_model
import cv2

img_rows = 140
img_cols = 140

model = load_model('transerlearning-flower-83.h5')
print("Loaded Succesfully.!")


test_image = cv2.imread('tuliplamp.jpg')
# test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
test_image = cv2.resize(test_image, (img_rows, img_cols))
test_image = np.array(test_image)
test_image = test_image.astype('float32')
test_image /= 255
print(test_image.shape)

test_image = np.expand_dims(test_image, axis=0)
print(test_image.shape)

prediction = model.predict(test_image)
print(np.argmax(prediction, axis=1))



