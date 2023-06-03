import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# https://www.tensorflow.org/tutorials/images/cnn
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
        plt.xlabel(class_names[train_labels[i][0]])
    plt.show()

def create_models():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))

    model.summary()
    return model

def train_models(model, train_images, train_labels, test_images, test_labels):
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
                  metrics=['accuracy'])
    history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels))
    return (model, history)
    

def evaluate_models(model, history, test_images, test_labels):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()

    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    print("test_acc", test_acc)
    print("test_loss", test_loss)

def main():
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    # create architecture
    model = create_models()
    # train data
    (trained_model, history) = train_models(model, train_images=train_images, train_labels=train_labels, test_images=test_images, test_labels=test_labels)
    # evaluate
    evaluate_models(model=trained_model, history=history, test_images=test_images, test_labels=test_labels)
    # save model
    trained_model.save('model.h5')
    

if __name__ == '__main__':
    main()