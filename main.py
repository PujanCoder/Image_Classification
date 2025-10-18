import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets , layers , models
from tensorflow.python.ops.metrics_impl import accuracy

(training_images, training_labels), (test_images, test_labels) = datasets.cifar10.load_data()
training_images = training_images / 255.0
test_images = test_images / 255.0
class_names = ['Plane','Car','Bird','Cat','Deer' ,'Dog','frog','Horse','Ship','Truck']
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.xticks([])

    plt.yticks([])
    plt.imshow(training_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[training_labels[i][0]])
plt.show()

training_images = training_images[:20000]
training_labels = training_labels[:20000]
test_images = test_images[:20000]
test_labels = test_labels[:20000]
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))

model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=10,validation_data=(test_images, test_labels))
loss, accuracy = model.evaluate(test_images, test_labels)
print('Test loss:', loss)
print('Test accuracy:', accuracy)
model.save('image_classifier.h5')




# To test the model we can input any image , and the model prediction name

img = cv.imread('dog.jpg', cv.IMREAD_GRAYSCALE)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

plt.imshow(img, cmap=plt.cm.binary)

prediction = model.predict(np.array([img])/255)
index = np.argmax(prediction)

print(prediction)
