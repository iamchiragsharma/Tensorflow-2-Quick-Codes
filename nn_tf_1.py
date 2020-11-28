import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

mnist = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape)

x_train, x_test = x_train/255.0, x_test/255.0

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape = (28,28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])

print(model.summary())


###### Another Way of Defining Model ######
# model = keras.Sequential()
# model.add(keras.layers.Flatten(input_shape = (28,28)))
# model.add(keras.layers.Dense(128, activation='relu'))
# model.add(keras.layers.Dense(10))
###########################################

loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optim = keras.optimizers.Adam(lr=0.001)
metrics = ["accuracy"]

model.compile(loss=loss, optimizer=optim, metrics=metrics)

batch_size = 64
epochs = 5

model.fit(x_train,y_train, batch_size, epochs=epochs, shuffle=True, verbose=2)

model.evaluate(x_test,y_test, batch_size, verbose=2)

probability_model = keras.models.Sequential([
    model,
    keras.layers.Softmax()
])



# By Numpy
# predictions = probability_model(x_test)
# labels = np.argmax(predictions, axis=1)
# print(labels.shape)


# By Tensorflow
predictions = model(x_test)
predictions = tf.nn.softmax(predictions)
labels = np.argmax(predictions, axis=1)

print(y_test[:5])
print(labels[:5])
