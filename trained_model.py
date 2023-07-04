import numpy as np
from tensorflow.keras import datasets, layers, models
import cv2 as cv
import matplotlib.pyplot as plt

# (training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()

# training_images, testing_images = training_images/255, testing_images/255

class_names= ["Plane", "Car", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]

# training_images = training_images[:20000]
# training_labels = training_labels[:20000]
# testing_images = testing_images[:5000]
# testing_labels = testing_labels[:5000]

model = models.load_model('image_classifier.model')
img = cv.imread('horse.jpg')


plt.imshow(img)
prediction = model.predict(np.array([img])/255)
index = np.argmax(prediction)

print(f"Prediction is {class_names[index]}")