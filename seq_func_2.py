#Sequential and Functional Api Keras
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


#Sequential API
model = keras.Sequential([
    layers.Flatten(input_shape = (28,28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10),
])

print(model.summary())

#Sequential Without List
seq_model = keras.Sequential()
seq_model.add(layers.Flatten(input_shape=(28,28)))
seq_model.add(layers.Dense(128, activation='relu'))
seq_model.add(layers.Dense(10))
print(seq_model.summary())


#Functional Model
inputs = layers.Input(shape=(28,28))
flatten = layers.Flatten()
dense1 = layers.Dense(128, activation='relu')
dense2 = layers.Dense(10)


x = flatten(inputs)
x = dense1(x)
outputs = dense2(x)

func_model = keras.Model(inputs = inputs, outputs = outputs, name = 'functional_api_model')

print(func_model.summary())


#Functional Model - Multiple Outputs
inputs = layers.Input(shape=(28,28))
flatten = layers.Flatten()
dense1 = layers.Dense(128, activation='relu')
dense2 = layers.Dense(10)
dense2_2 = layers.Dense(1)

x = flatten(inputs)
x = dense1(x)

outputs = dense2(x)
outputs2_2 = dense2_2(x)

func_model_2 = keras.Model(inputs = inputs, outputs = [outputs, outputs2_2], name = 'functional_api_model')

print(func_model_2.summary())

#### Accessing Layers in Functional Model
inputs = func_model_2.input
outputs = func_model_2.output

#### Acccesing Inputs and Outputs of a Particular Layer
input0 = func_model_2.layers[0].input
output0 = func_model_2.layers[0].output


print(inputs)
print(outputs)


print(input0)
print(output0)



### Transfer Learning via Functional API

base_model = keras.applications.VGG16()
print(base_model.summary())

x = base_model.layers[-2].output
new_outputs = layers.Dense(1)(x)

new_model = keras.Model(inputs=base_model.inputs, outputs = new_outputs, name = 'transfer_learning_model')

print(new_model.summary())