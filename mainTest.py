import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

model = load_model('BrainTumor10Epochs.h5')

image = cv2.imread("C:\\Users\\rampr\\Desktop\\brain tumor\\pred\\pred6.jpg")

img = Image.fromarray(image)

img = img.resize((64, 64))

img = np.array(img)

input_img = np.expand_dims(img, axis=0)

# Use predict to get raw predictions
predictions = model.predict(input_img)

# Post-process predictions to get the predicted class
predicted_class = np.argmax(predictions)

print("Predicted class:", predicted_class)