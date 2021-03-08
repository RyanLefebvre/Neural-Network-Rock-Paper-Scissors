import cv2
import numpy as np
from keras_squeezenet import SqueezeNet
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.layers import Activation, Dropout, Convolution2D, GlobalAveragePooling2D
from keras.models import Sequential
import tensorflow as tf
import os

# Name of the folder that contains the training dataset.
imageDirectoryName = 'training_images'

# Possible classes for images.
imageClasses = {
    "rock": 0,
    "paper": 1,
    "scissors": 2,
    "none": 3
}

# Number of classes ~ 4.
NUM_CLASSES = len(imageClasses)

# Returns the numeric value associated with one of the
# possible image classes for rock, paper, scissors or none.
def getClassValue(val):
    return imageClasses[val]

# Returns the model that will be trained with images of hand.
def get_model():
    model = Sequential([
        SqueezeNet(input_shape=(250, 250, 3), include_top=False),
        Dropout(0.5),
        Convolution2D(NUM_CLASSES, (1, 1), padding='valid'),
        Activation('relu'),
        GlobalAveragePooling2D(),
        Activation('softmax')
    ])
    return model


# Load images, train the model and save to file.
dataset = []
for directory in os.listdir(imageDirectoryName):
    path = os.path.join(imageDirectoryName, directory)
    if not os.path.isdir(path):
        continue
    for item in os.listdir(path):
        if item.startswith("."):
            continue
        img = cv2.imread(os.path.join(path, item))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (250, 250))
        dataset.append([img, directory])

data, labels = zip(*dataset)
labels = list(map(getClassValue, labels))
labels = np_utils.to_categorical(labels)
model = get_model()
model.compile(
    optimizer=Adam(lr=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(np.array(data), np.array(labels), epochs=10)
model.save("model.h5")
