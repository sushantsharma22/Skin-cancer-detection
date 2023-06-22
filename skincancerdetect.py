import os
import numpy as np
import cv2 as cv
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# Function to read, resize and import data into a list
def input_data(folder_path, output_data):
    for dirs in os.listdir(folder_path):
        class_name = dirs
        new_path = os.path.join(folder_path, class_name)
        for img in os.listdir(new_path):
            img_arr = cv.imread(os.path.join(new_path, img))
            resized = cv.resize(img_arr, (64, 64))   # reducing image size for efficiency
            output_data.append([resized, class_name])
    return output_data

# Importing the training and testing data
train_data = input_data("D:/python/skin cancer detection/train", [])
test_data = input_data("D:/python/skin cancer detection/test", [])

# Shuffling the data
np.random.shuffle(train_data)
np.random.shuffle(test_data)

# Separating the images and labels
train_images, train_labels = zip(*train_data)
test_images, test_labels = zip(*test_data)

# Encoding the labels
label_enc = LabelEncoder()
train_labels = label_enc.fit_transform(train_labels)
test_labels = label_enc.transform(test_labels)

# Convert images and labels to arrays
train_images = np.array(train_images)
train_labels = np.array(train_labels)
test_images = np.array(test_images)
test_labels = np.array(test_labels)

# Normalizing the image pixels
train_images = train_images / 255.0
test_images = test_images / 255.0

# Convert labels to categorical (one-hot encoding)
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Data Augmentation
datagen = ImageDataGenerator(rotation_range=20, zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15, horizontal_flip=True, fill_mode="nearest")

# Load VGG16 model with pre-trained weights from ImageNet without the top layer (fully connected layers)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))

# Adding a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)

# Adding a fully-connected layer
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)

# Adding a logistic layer with number of classes (assuming you have 2 classes: malignant and benign)
predictions = Dense(2, activation='softmax')(x)

# Model to be trained
model = Model(inputs=base_model.input, outputs=predictions)

# Compile model
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)

# Training model
history = model.fit(datagen.flow(train_images, train_labels, batch_size=32), validation_data=(test_images, test_labels), steps_per_epoch=len(train_images) / 32, epochs=100, verbose=1, callbacks=[es])

# Print accuracy
accuracy = history.history['val_accuracy'][-1]
print('Validation Accuracy:', accuracy)
