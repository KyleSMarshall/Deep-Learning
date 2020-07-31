'''
Boeing requested a model that will anaylze greyscale images of airplane fusilages and be capable of detecting corrosion.

The model should return a 1 if it find corrosion, and a 0 if it does not.

This project was special in that there was a very small amount of data provided (~250 images split evenly into corroded and non-corroded classes).

The solution employed transfer learning and realtime image augmentation in order to produce a model with accurate results and that could generalize 
well to unseen data.

The results are being written into a paper at the moment and will be published soon, so the entierty of the project code will not be shown below,
instead, I will provide code for the runner-up model, DenseNet121.
'''

#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import the required deep-learning modules
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.xception import Xception, preprocess_input, decode_predictions
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input, decode_predictions
from tensorflow.keras.applications.densenet import DenseNet201, DenseNet121, preprocess_input, decode_predictions
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
from keras.layers import LeakyReLU
import keras.backend as K

# Import other modules
import pandas as pd
import numpy as np
import random
import os
import cv2

# Deifne the desired width and height of the image (~95% of the images had this dimension to begin with. Setting standard to scale the others.)
width = 480
height = 640
channels = 1

X_names, y_train = [], []

# Obtain image names and paths and append to an array
# Beginning first with the corroded images
files = os.listdir("corrosion_analysis/corrosion/") 
for i in range(len(files)):
  X_names.append("corrosion_analysis/corrosion/" + files[i])
  y_train.append(1)                   # Append 1 indicating corrosion

files = os.listdir("corrosion_analysis/none/")
for i in range(len(files)):
  X_names.append("corrosion_analysis/none/" + files[i])
  y_train.append(0)                   # Append 0 indicating no corrosion
    
    
y_train = np.array(y_train)           # Convert to a numpy array
X_names, y_train = shuffle(X_names, y_train, random_state=11)           # Shuffle the data

# Split the data into k-folds and return the indicies for the data in each fold.
k = 5         # Number of folds used for cross-validation of model
folds = list(StratifiedKFold(n_splits=k).split(X_names, y_train))

'''
We now need to load in the actual image data from the path names stored in X_names.
For this we will use cv2.imread()
We begin by initializing a zeros array with the required size for storing the image data.
'''
X_train = np.zeros((len(X_names), width, height, 3))

# Iteratively load in each image and store its numerical data in the X_train array
for i in range(len(X_names)):
    X = cv2.imread(X_names[i])
    X = cv2.resize(X, (height, width), interpolation = cv2.INTER_AREA)        # Resize using the INTER_AREA algorithm if resizing is required
    
    X_train[i] = X

del X_names                # Clear up memory used to store variable


'''
Now we will define the image data generators that will perform realtime image augmentation during training.

This image augmentation adds variance to the dataset and helps the model to recognize features that are irrelevant to the
classification task at hand. We will introduce random zooms, x and y axis shifts, horrizontal and vertical image flips,
and rotations up to 20 degrees. The fill mode for the process will be constant black (value of 0).
'''
datagen = ImageDataGenerator(rotation_range=20,
                             width_shift_range=0.15,
                             height_shift_range=0.15,
                             zoom_range=0.15,
                             horizontal_flip=True,
                             vertical_flip=True, 
                             fill_mode='constant')
                             
# Define a custom model validation metric known as the f1_score                            
def get_f1(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val
   
# We want to reduce the learning rate when we hit a plateau for 15 epochs
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    verbose=0,
    mode="auto",
    patience=15,
    min_lr=0
)
    
# Stop the training if we go 55 epochs without a decrease in the validation set loss function
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=55,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=True,
)

# Define out callback list
callback_list = [reduce_lr, early_stop]


# Model
'''
Here is where the model is implemented.

We begin with a DenseNet121 framework with weights pretrained on the imagenet dataset.
This model will have good low-level feature extraction methods and through fine tuning using
this dataset, will be able to extract high-level features specific to the classification task.

The classifier built on top of the DenseNet121 architecture is a fully connected dense layer
with 1028 neurons. Activation is leakyrelu to avoid sigmoid saturation.

Batch normalization is used to increase learning speed and improve overall performance.
Dropout is used to help the model generalize well to unseen data.
'''
conv_base = DenseNet121(include_top=False, weights='imagenet', input_shape=(width, height, 3))

model = Sequential()
model.add(conv_base)

model.add(Flatten())

model.add(Dense(units=1028))
model.add(LeakyReLU(alpha=0.2))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(units=1, activation='sigmoid'))

model.summary()

'''
After the model is created we will save the weights to disk. This is what will allow us to 
effectively perform k-fold cross validation without the risk of data leakage.
'''
model.save_weights('model_121.h5')

# Use Adam optimizer and compile to model
opt = tf.optimizers.Adam(learning_rate=0.00005)
model.compile(optimizer=opt, loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), get_f1])


# Initalize history variable so that results can be easily accessed after training
history = []

# First fold
j = 1
# Obtain train and validation indices for this fold
train_idx, val_idx = folds[j-1]
print('\nFold ',j)

# Load in the training and validation data for this fold
X_train_cv = X_train[train_idx]
y_train_cv = y_train[train_idx]
X_valid_cv = X_train[val_idx]
y_valid_cv= y_train[val_idx]

# Use the image data generator to stream the dataset to the model
train_generator = datagen.flow(
    X_train_cv,
    y_train_cv, 
    batch_size=4,
    seed=11)

# Train the model for 200 epochs
model.fit(train_generator,
          epochs=200,
          shuffle=True,
          verbose=1,
          validation_data = (X_valid_cv, y_valid_cv),
          callbacks=callback_list)

temp_history = model.evaluate(X_valid_cv, y_valid_cv)
history.append(temp_history)
print(model.evaluate(X_valid_cv, y_valid_cv))
    

    
# Fold 2
j = 2
train_idx, val_idx = folds[j-1]
print('\nFold ',j)

# Here we load in the weights that were saved immediately after the model was created and had not seen any training examples
model.load_weights('model_121.h5')

opt = tf.optimizers.Adam(learning_rate=0.00005)
model.compile(optimizer=opt, loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), get_f1])

X_train_cv = X_train[train_idx]
y_train_cv = y_train[train_idx]
X_valid_cv = X_train[val_idx]
y_valid_cv= y_train[val_idx]

train_generator = datagen.flow(
    X_train_cv,
    y_train_cv, 
    batch_size=4,
    seed=11)

model.fit(train_generator,
          epochs=200,
          shuffle=True,
          verbose=1,
          validation_data = (X_valid_cv, y_valid_cv),
          callbacks=callback_list)

temp_history = model.evaluate(X_valid_cv, y_valid_cv)
history.append(temp_history)
print(model.evaluate(X_valid_cv, y_valid_cv))


# Fold 3
j = 3
train_idx, val_idx = folds[j-1]
print('\nFold ',j)

model.load_weights('model_121.h5')

opt = tf.optimizers.Adam(learning_rate=0.00005)
model.compile(optimizer=opt, loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), get_f1])


X_train_cv = X_train[train_idx]
y_train_cv = y_train[train_idx]
X_valid_cv = X_train[val_idx]
y_valid_cv= y_train[val_idx]

train_generator = datagen.flow(
    X_train_cv,
    y_train_cv, 
    batch_size=4,
    seed=11)

model.fit(train_generator,
          epochs=200,
          shuffle=True,
          verbose=1,
          validation_data = (X_valid_cv, y_valid_cv),
          callbacks=callback_list)

temp_history = model.evaluate(X_valid_cv, y_valid_cv)
history.append(temp_history)
print(model.evaluate(X_valid_cv, y_valid_cv))

# Fold 4
j = 4
train_idx, val_idx = folds[j-1]
print('\nFold ',j)

model.load_weights('model_121.h5')

opt = tf.optimizers.Adam(learning_rate=0.00005)
model.compile(optimizer=opt, loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), get_f1])



X_train_cv = X_train[train_idx]
y_train_cv = y_train[train_idx]
X_valid_cv = X_train[val_idx]
y_valid_cv= y_train[val_idx]

train_generator = datagen.flow(
    X_train_cv,
    y_train_cv, 
    batch_size=4,
    seed=11)

model.fit(train_generator,
          epochs=200,
          shuffle=True,
          verbose=1,
          validation_data = (X_valid_cv, y_valid_cv),
          callbacks=callback_list)

temp_history = model.evaluate(X_valid_cv, y_valid_cv)
history.append(temp_history)
print(model.evaluate(X_valid_cv, y_valid_cv))

# Fold 5
j = 5
train_idx, val_idx = folds[j-1]
print('\nFold ',j)

model.load_weights('model_121.h5')


opt = tf.optimizers.Adam(learning_rate=0.00005)
model.compile(optimizer=opt, loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), get_f1])


X_train_cv = X_train[train_idx]
y_train_cv = y_train[train_idx]
X_valid_cv = X_train[val_idx]
y_valid_cv= y_train[val_idx]

train_generator = datagen.flow(
    X_train_cv,
    y_train_cv, 
    batch_size=4,
    seed=11)

model.fit(train_generator,
          epochs=200,
          shuffle=True,
          verbose=1,
          validation_data = (X_valid_cv, y_valid_cv),
          callbacks=callback_list)

temp_history = model.evaluate(X_valid_cv, y_valid_cv)
history.append(temp_history)
print(model.evaluate(X_valid_cv, y_valid_cv))


'''
After the 5-fold training is complete, the results are then compiled into accessible arrays
so that the statistics can be determined.
'''
val_loss = np.array([history[0][0], history[1][0], history[2][0], history[3][0], history[4][0]])
val_accuracy = np.array([history[0][1], history[1][1], history[2][1], history[3][1], history[4][1]])
val_precision = np.array([history[0][2], history[1][2], history[2][2], history[3][2], history[4][2]])
val_recall = np.array([history[0][3], history[1][3], history[2][3], history[3][3], history[4][3]])
val_f1 = np.array([history[0][4], history[1][4], history[2][4], history[3][4], history[4][4]])


'''
Calculate mean and standard deviation for the metrics.
Using these statistics and t-statistics, the 95% confidence interval for each metric was calculated
and used to analyze and compare the model's performance agaisnt others.
'''
val_loss_mean = np.mean(val_loss)
val_loss_std = np.std(val_loss)
print("loss: ",val_loss_mean, val_loss_std)

val_accuracy_mean = np.mean(val_accuracy)
val_accuracy_std = np.std(val_accuracy)
print("accuracy: ", val_accuracy_mean, val_accuracy_std)

val_precision_mean = np.mean(val_precision)
val_precision_std = np.std(val_precision)
print("precision: ", val_precision_mean, val_precision_std)

val_recall_mean = np.mean(val_recall)
val_recall_std = np.std(val_recall)
print("recall: ", val_recall_mean, val_recall_std)

val_f1_mean = np.mean(val_f1)
val_f1_std = np.std(val_f1)
print("f1_score: ", val_f1_mean, val_f1_std)
