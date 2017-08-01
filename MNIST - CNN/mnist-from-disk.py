
from scipy.misc import imread
from keras.models import load_model


model = load_model('mnistmodel.h5')
print("Loaded Succesfully.!")

image = imread("img59919.jpg")
image = image / 255
prediction = model.predict_classes(image.reshape((1, 1, 28, 28)))
print(prediction)