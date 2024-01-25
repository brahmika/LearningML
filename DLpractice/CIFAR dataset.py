import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import datasets, models, layers
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images = train_images/255.0
test_images = test_images/255.0
class_names = ['airplane', 'automobile', 'bird','cat','deer','dog','frog','horse','ship','truck']

img_index = 1
plt.imshow(train_images[img_index], cmap=plt.cm.binary)
plt.xlabel(class_names[train_labels[img_index][0]])
plt.show()

model = models.Sequential()
model.add(layers.Conv2D(32,(3,3), activation = 'relu',input_shape = (32,32,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3), activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,((2,2)), activation = 'relu'))

model.summary()

#Adding Dense Layers
model.add(layers.Flatten())
model.add(layers.Dense(64, activation = 'relu'))
model.add(layers.Dense(10))

model.summary()

#Training data
model.compile(optimizer = 'adam',
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
              metrics = ['accuracy'])
history = model.fit(train_images, train_labels, epochs = 4, validation_data = (test_images, test_labels))

#Evaluating the data
test_loss , test_acc = model.evaluate(test_images, test_labels, verbose = 2)
print(test_acc)
