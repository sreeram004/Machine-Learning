import numpy as np
from keras.models import load_model
import cv2

img_rows = 32
img_cols = 32

model = load_model('cnnmodelextplotadamfromweight72.h5')
print("Loaded Succesfully.!")

test_image = cv2.imread('tulips01.jpg')
# test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
test_image = cv2.resize(test_image, (img_rows, img_cols))
test_image = np.array(test_image)
test_image = test_image.astype('float32')
test_image /= 255
print(test_image.shape)

test_image = np.rollaxis(test_image, 2, 0)
test_image = np.expand_dims(test_image, axis=0)
print(test_image.shape)

prediction = model.predict_classes(test_image)
print(prediction)
