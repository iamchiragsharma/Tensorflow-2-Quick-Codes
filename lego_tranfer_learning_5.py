import os

from tensorflow.python.keras.applications.imagenet_utils import preprocess_input
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import tensorflow.keras as keras

import matplotlib.pyplot as plt


CLASS_NAMES = ['YODA', 'SKY WALKER', 'R2-D2', 'MACE WINDU', 'GENERAL GRIEVIOUS', 
'DARTH VADER', 'RETARD VADER', 'CHICK WITH GUN', 'ZOMBIE BOBBAFETT','DARTH VADER WITHOUT MASK', 'POC']

preprocess_input = keras.applications.vgg16.preprocess_input

train_gen = keras.preprocessing.image.ImageDataGenerator(preprocessing_function= preprocess_input)
test_gen = keras.preprocessing.image.ImageDataGenerator(preprocessing_function= preprocess_input)
valid_gen = keras.preprocessing.image.ImageDataGenerator(preprocessing_function= preprocess_input)

train_batches = train_gen.flow_from_directory(
    directory= os.path.join(*["lego_classification", "lego_dataset_standard", "train"]),
    target_size= (256,256),
    class_mode = 'sparse',
    batch_size= 4,
    shuffle= True,
    color_mode= 'rgb',
    classes = CLASS_NAMES
)

test_batches = train_gen.flow_from_directory(
    directory= os.path.join(*["lego_classification", "lego_dataset_standard", "test"]),
    target_size= (256,256),
    class_mode = 'sparse',
    batch_size= 4,
    shuffle= False,
    color_mode= 'rgb',
    classes = CLASS_NAMES
)

val_batches = train_gen.flow_from_directory(
    directory= os.path.join(*["lego_classification", "lego_dataset_standard", "val"]),
    target_size= (256,256),
    class_mode = 'sparse',
    batch_size= 4,
    shuffle= False,
    color_mode= 'rgb',
    classes = CLASS_NAMES
)


vgg_model = keras.applications.vgg16.VGG16()
print(type(vgg_model)) # > Keras Functional Model
model = keras.models.Sequential()

#Converting to Sequential Model
for layer in vgg_model.layers[:-1]:
    model.add(layer)

print(model.summary())


for layer in model.layers:
    layer.trainable = False


model.add(keras.layers.Dense(11))


loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = keras.optimizers.Adam(0.001)
metrics = ['accuracy']

model.compile(loss=loss, optimizer=optimizer, metrics=metrics)


epochs = 20

#early stopping
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, verbose = 2)

history = model.fit(train_batches, validation_data=val_batches, epochs=epochs, verbose=2)

model.save(os.path.join(*["lego_classification", "lego_classification_model.h5"]))


#Plotting losses
plt.figure(figsize=(15,6))
plt.subplot(1,2,1)
plt.plot(history.history["loss"], label="train loss")
plt.plot(history.history["val_loss"], label="valid loss")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history["accuracy"], label="train_acc")
plt.plot(history.history["val_accuracy"], label="valid_acc")
plt.legend()

plt.show()



model.evaluate(test_batches, verbose=2)


predictions = model.predict(test_batches)
predictions = tf.nn.softmax(predictions, axis=1)
print(predictions)