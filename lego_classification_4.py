import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import tensorflow.keras as keras

import matplotlib.pyplot as plt


CLASS_NAMES = ['YODA', 'SKY WALKER', 'R2-D2', 'MACE WINDU', 'GENERAL GRIEVIOUS', 
'DARTH VADER', 'RETARD VADER', 'CHICK WITH GUN', 'ZOMBIE BOBBAFETT','DARTH VADER WITHOUT MASK', 'POC']

train_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255)
test_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255)
valid_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255)

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


model = keras.models.Sequential()
model.add(keras.layers.Conv2D(32, (3,3), strides=(1,1), padding='valid', activation='relu', input_shape= (256,256,3)))
model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.Conv2D(64, (3,3), activation='relu'))
model.add(keras.layers.MaxPooling2D(2,2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(11))

print(model.summary())


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