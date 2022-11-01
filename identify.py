from matplotlib import pyplot
import tensorflow as tf
from tensorflow import keras
import numpy as np
import ssl

#hack to get around ssl error
ssl._create_default_https_context = ssl._create_unverified_context

#fashion = keras.datasets.fashion_mnist
nums = keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = nums.load_data()

#we get the shapes of the vectors
print('X_train: ' + str(train_images.shape))
print('Y_train: ' + str(train_labels.shape))
print('X_test:  '  + str(test_images.shape))
print('Y_test:  '  + str(test_labels.shape))


for i in range(9):
    pyplot.subplot(330 + 1 + i)
    pyplot.imshow(train_images[i], cmap=pyplot.cm.binary)
    pyplot.xlabel(train_labels[i])


pyplot.show()

#we normalize the data
train_images = train_images / 255.0
test_images = test_images / 255.0

#we create the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.sigmoid),
    keras.layers.Dense(128, activation=tf.nn.sigmoid),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

#we compile the model
model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

#we train the model
model.fit(train_images, train_labels, epochs=10)

#we test the model
test_loss, test_acc = model.evaluate(test_images, test_labels)

#display 9 images with their predictions
predictions = model.predict(test_images)
for i in range(9):
    pyplot.subplot(330 + 1 + i)
    pyplot.imshow(test_images[i], cmap=pyplot.cm.binary)
    pyplot.xlabel(np.argmax(predictions[i]))

pyplot.show()

print('Test accuracy:', test_acc)

