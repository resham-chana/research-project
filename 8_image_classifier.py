# import relevant libraries
import matplotlib.pyplot as plt
import numpy as np
import setuptools.dist
import tensorflow as tf
import os
from tensorflow.keras.utils import image_dataset_from_directory 
from tensorflow.keras.models import Model 
from tensorflow import keras
from tensorflow.keras import layers 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout 
from sklearn.metrics import confusion_matrix, fbeta_score
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# initialise batch size and image size and load in data 
BATCH_SIZE = 32 
IMG_SIZE = (128, 128)
train_dir = "C:/Users/resha/images/train"

train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir,
                                             labels='inferred',  # labels from directory
                                             batch_size = BATCH_SIZE,
                                             image_size=IMG_SIZE)
val_dir = 'C:/Users/resha/images/val'

val_dataset = tf.keras.utils.image_dataset_from_directory(val_dir,
                                             labels='inferred',  # labels from directory 
                                             batch_size = BATCH_SIZE,
                                             image_size=IMG_SIZE)

test_dir = 'C:/Users/resha/images/test'


test_dataset = tf.keras.utils.image_dataset_from_directory(test_dir,
                                             labels='inferred',  # labels from directory 
                                             batch_size = BATCH_SIZE,
                                             image_size=IMG_SIZE)

# Allocate GPU memory as needed
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# code adapted from [https://www.tensorflow.org/tutorials/images/transfer_learning]

class_names = train_dataset.class_names
# plot the first 9 images from the train dataset
plt.style.use("dark_background")
plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
  for i in range(15):
    ax = plt.subplot(5, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
plt.suptitle("First 15 Images from the Training Dataset", fontsize=16)
plt.savefig()
plt.show()

# check the shape for a random image to check it has been resized
for image_batch, label_batch in train_dataset.take(1):
    # Get the shape of the image
    image_shape = image_batch[0].shape
    # Print the shape of the image
    print("Pixel length:", image_shape[0])
    print("Pixel height:", image_shape[1])


def preprocess_images(dataset):
    images = []
    labels = []
    for image_batch, label_batch in dataset:
        for img, lbl in zip(image_batch.numpy(), label_batch.numpy()):
            images.append(img.flatten())  # Flatten the image
            labels.append(lbl)
    return np.array(images), np.array(labels)

# Preprocess training images

# using buffered prefetching 
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

# augment data to have some horizontal flips
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip('horizontal')
])

# Apply the rescaling to the input
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

# base model from pre-trained model
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

# get the feature extractor and shape
image_batch, label_batch = next(iter(train_dataset))
feature_batch = base_model(image_batch)
print(feature_batch.shape)

# Freeze convolutional base 
base_model.trainable = False

# set the pooling method
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)

#global_max_layer = tf.keras.layers.GlobalMaxPooling2D()
#feature_batch_max = global_max_layer(feature_batch)
#print(feature_batch_max.shape)

prediction_layer = tf.keras.layers.Dense(15, activation='softmax')
# pass features through the output layer
prediction_batch = prediction_layer(feature_batch_average)

# shape of the prediction batch
print(prediction_batch.shape)


# define input shape and apply data sugmentation, rescaling and pass this through the model
inputs = tf.keras.Input(shape=(128, 128, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_average_layer(x)
# also experimented with no Dropout
x = tf.keras.layers.Dropout(0.2)(x)

outputs = prediction_layer(x)

model = tf.keras.Model(inputs, outputs)
model.summary()


# compile model and set learning rate
base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')])



# set epochs and evaluate loss and accuracy
initial_epochs = 10
loss0, accuracy0 = model.evaluate(val_dataset)

print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))

# fit the model and train
history = model.fit(train_dataset,
                    epochs=initial_epochs,
                    validation_data=val_dataset)

                    
# extracting accuracy and loss from model.fit
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

# plotting these across epochs
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy',color="red")
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),0.5])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss', color="red")
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,3])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.savefig("test_and_training1.png")
plt.show()

loss, accuracy = model.evaluate(val_dataset)
print("Validation loss: {:.2f}".format(loss))
print("Validation accuracy: {:.2f}".format(accuracy))

#Validation loss: 2.26
#Validation accuracy: 0.28

########### Fine Tuning ##########

# Unfreeze base model and set bottom layers as untrainable
base_model.trainable = True

# Number of layers in the base model
print("Number of layers in the base model: ", len(base_model.layers))

# Also tested: fine_tune_at = 100
fine_tune_at = 120

# freeze layers prior to layer 120
for layer in base_model.layers[:fine_tune_at]:
  layer.trainable = False

# recompile model with a lower learning rate 
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate/10),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')])

model.summary()

fine_tune_epochs = 10
total_epochs =  initial_epochs + fine_tune_epochs

history_fine = model.fit(train_dataset,
                         epochs=total_epochs,
                         initial_epoch=history.epoch[-1],
                         validation_data=val_dataset)


# redefine accuracy and loss as the summation of both models
acc = history.history['accuracy'] + history_fine.history['accuracy']
val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']
loss = history.history['loss'] + history_fine.history['loss']
val_loss = history.history['val_loss'] + history_fine.history['val_loss']

# plot this
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0.8, 1])
plt.plot([initial_epochs-1,initial_epochs-1],
          plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 1.0])
plt.plot([initial_epochs-1,initial_epochs-1],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

#plt.savefig("fine_tune")
# print final test accuracy
loss, accuracy = model.evaluate(test_dataset)
print('Test accuracy:', accuracy)


# get a batch of images from the test set
image_batch, label_batch = test_dataset.as_numpy_iterator().next()
predictions = model.predict_on_batch(image_batch)

# apply softmax activation to convert to probabilities 
predictions = tf.nn.softmax(predictions)

# predicted classes:
predicted_classes = tf.argmax(predictions, axis=1)

print('Predicted Classes:\n', predicted_classes.numpy())
print('True Classes:\n', label_batch)

# plot this
plt.figure(figsize=(10, 10))
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image_batch[i].astype("uint8"))
    plt.title(class_names[predicted_classes[i]])  
    plt.axis("off")
plt.show()

# create a confusion matrix 

all_predicted_classes = []
all_true_classes = []

# iterate over all batches 
for image_batch, label_batch in test_dataset.as_numpy_iterator():
    # predict classification for batch 
    predictions = model.predict_on_batch(image_batch)
    
    # convert logits to probabilities
    predictions = tf.nn.softmax(predictions)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # append predicted and true classes 
    all_predicted_classes.extend(predicted_classes)
    all_true_classes.extend(label_batch)

# calculate confusion matrix
conf_matrix = confusion_matrix(all_true_classes, all_predicted_classes)
print("Confusion Matrix:")
print(conf_matrix)

# plot the confusion matrix
plt.figure(figsize=(10, 8))
plt.imshow(conf_matrix, cmap='viridis', interpolation='nearest')
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# calculate f1 score 

beta = 1  
fbeta = fbeta_score(all_true_classes, all_predicted_classes, beta=beta, average='weighted')
print("F1 score: {:.4f}".format(fbeta))