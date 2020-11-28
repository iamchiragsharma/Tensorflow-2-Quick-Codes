import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

import numpy as np
import matplotlib.pyplot as plt


#Let's Predict Digit as well the orientation of digit

inputs = keras.layers.Input(shape = (28,28))
flatten = keras.layers.Flatten()

dense1 = keras.layers.Dense(128, activation='relu')
dense2 = keras.layers.Dense(10, activation='softmax', name='category_output')
dense3 = keras.layers.Dense(1, activation='sigmoid', name='hand_orientation_output')

x = flatten(inputs)
x = dense1(x)
category = dense2(x)
hand = dense3(x)

model = keras.Model(inputs = inputs, outputs= [category, hand], name='complete-mnist-model')
print(model.summary())


category_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
hand_loss = keras.losses.BinaryCrossentropy(from_logits=False)
optim = keras.optimizers.Adam(0.001)
metrics = ['accuracy']

losses = {
    'category_output' : category_loss,
    'hand_orientation_output' : hand_loss
    }

#Model Compilation
model.compile(optimizer=optim, metrics=metrics, loss=losses)


#Dataset
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0

y_orientation = np.zeros(y_train.shape, dtype=np.uint8)
for idx,y in enumerate(y_train):
    if y > 5: #Right Hand
        y_orientation[idx] = 1

y = {
    'category_output' : y_train,
    'hand_orientation_output' : y_orientation
}

## Test Output
print("Test Input - Output")
print(y_train[:16])
print(y_orientation[:16])


#Training
model.fit(x_train, y=y, epochs=5, batch_size=64, verbose=1)


predictions = model.predict(x_test)
prediction_category = np.argmax(predictions[0],1)
prediction_orientation = [0 if p < 0.5 else 1 for p in predictions[1]]

print(f"Original Categories : {y_test[:20]}")
print(f"Predicted Categories : {prediction_category[:20]}")

original_orientations = [0 if p < 0.5 else 1 for p in y_test[:20]]
print(f"Original Orientation : {prediction_orientation[:20]}")
print(f"Predicted Orientation : {prediction_orientation[:20]}")