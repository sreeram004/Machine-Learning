from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.applications.inception_v3 import decode_predictions
from keras.preprocessing import image as image_utils
import cv2
from keras import backend as K
import numpy as np

K.set_image_dim_ordering('tf')

img_rows = 299
img_cols = 299

disp_rows = 500
disp_cols = 500

model = InceptionV3(weights='imagenet')

image = image_utils.load_img('violin.jpg', target_size=(img_rows, img_cols))
image = image_utils.img_to_array(image)


# Convert (3, 299, 299) to (1, 3, 299, 299)
# Here "1" is the number of images passed to network
# We need it for passing batch containing serveral images in real project

image = np.expand_dims(image, axis=0)
image = preprocess_input(image)

prediction = model.predict(image)


label = decode_predictions(prediction, top=1)[0]

orig = cv2.imread('violin.jpg')
orig = cv2.resize(orig, (disp_rows, disp_cols))


# Display the predictions
print('Prediction : ', label)
cv2.putText(orig, "Label: {}".format(label), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
cv2.imshow("Classification", orig)
cv2.waitKey(0)
