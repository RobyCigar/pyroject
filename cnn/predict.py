from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
import math


def plot(train_labels, train_images):
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    plt.figure(figsize=(10, 10))
    for i in range(36):
        plt.subplot(6, 6, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(True)
        plt.imshow(train_images[i])
        # The CIFAR labels happen to be arrays,
        # which is why you need the extra index
        print(math.floor(train_labels[i][0]))
        plt.xlabel(int(math.floor(train_labels[i][0])))
    plt.show()

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
model = keras.models.load_model('cnn/model.h5')
prediction = model.predict(test_images)
print(prediction)
plot(prediction, test_images)