# Basics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time

# ML
import tensorflow as tf
from tensorflow.keras.layers import *

# Miscellaneous
import warnings
warnings.filterwarnings("ignore")

# ML
import tensorflow as tf
import tensorflow.keras.backend as K

# Modelling
from tensorflow.keras.layers import *

ROOT = "./Vegetable Images"
CLASSES = ['Broccoli','Capsicum','Bottle_Gourd','Radish','Tomato','Brinjal','Pumpkin','Carrot','Papaya','Cabbage','Bitter_Gourd','Cauliflower','Bean','Cucumber','Potato']
tfClasses = tf.constant(CLASSES)
TRAIN = tf.data.Dataset.list_files(f"{ROOT}/train/*/*")
TEST = tf.data.Dataset.list_files(f"{ROOT}/test/*/*")
VAL = tf.data.Dataset.list_files(f"{ROOT}/validation/*/*")
dataCats = ["train","test","validation"]

classCounts = []
for cat in dataCats:
    class_distro = [len(tf.data.Dataset.list_files(f"{ROOT}/{cat}/{i}/*")) for i in CLASSES]
    classCounts.append(class_distro)
    
train_distro = classCounts[0]

targetSize = max(train_distro)
additional_needed = [targetSize-i for i in train_distro]

aug_train = []
for i,v in enumerate(CLASSES):
    path = f'{ROOT}/train/{v}'
    imgNeeded = additional_needed[i]
    images = os.listdir(path)[:imgNeeded]
    aug_train.extend([path + "/" + i for i in images])
train_data_aug = tf.data.Dataset.from_tensor_slices(aug_train)

def createPreprocessor(imgSize):
    def processing(path):
        label = tf.strings.split(path , os.path.sep)
        one_hot = label[-2] == tfClasses
        label = tf.argmax(one_hot)
        label = tf.one_hot(label, depth=len(CLASSES))
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [imgSize, imgSize])
        img = tf.image.rgb_to_grayscale(img)
        img = img / 255.0
        return img , label
    return processing
    
def augmentation(image,label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.rot90(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    return image , label

def configure_for_performance(ds):
  ds = ds.cache()
  ds = ds.shuffle(buffer_size=1000)
  ds = ds.batch(32)
  ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
  return ds

train_ori_large = TRAIN.map(createPreprocessor(128))
train_aug_large = TRAIN.map(createPreprocessor(128)).map(augmentation)
train_large_ds = train_ori_large.concatenate(train_aug_large)

test_large_ds = TEST.map(createPreprocessor(128))
val_large_ds = VAL.map(createPreprocessor(128))

data = train_large_ds.concatenate(test_large_ds).concatenate(val_large_ds)
data = configure_for_performance(data)

train_ori_small = TRAIN.map(createPreprocessor(31))
train_aug_small = TRAIN.map(createPreprocessor(31)).map(augmentation)
train_small_ds = train_ori_small.concatenate(train_aug_small)

test_small_ds = TEST.map(createPreprocessor(31))
val_small_ds = VAL.map(createPreprocessor(31))

data_small = train_small_ds.concatenate(test_small_ds).concatenate(val_small_ds)
data_small = configure_for_performance(data_small)

reg_cnn_model = tf.keras.Sequential([
    Conv2D(16, 3, activation='relu', input_shape=(31, 31, 1)),
    BatchNormalization(),
    MaxPooling2D(),
    
    Conv2D(32, 3, activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),
    
    Conv2D(64, 3, activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),
    
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5), 
    Dense(15, activation='softmax')
])

reg_cnn_model_large = tf.keras.Sequential([
    Conv2D(32, 3, activation='relu', input_shape=(128, 128, 1)),
    BatchNormalization(),
    MaxPooling2D(),
    
    Conv2D(64, 3, activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),
    
    Conv2D(128, 3, activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),
    
    Conv2D(256, 3, activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),
    
    Flatten(),
    
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5), 
    
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5), 
    
    Dense(15, activation='softmax')
])

es = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=1, min_lr=1e-6)


reg_cnn_model.compile(optimizer='adam', 
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    
modelHist = reg_cnn_model.fit(
      data_small, epochs=25, callbacks=[es,reduce_lr]
    )

reg_cnn_model.save("./cnn_small/1", save_format="tf")



reg_cnn_model_large.compile(optimizer='adam', 
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    
modelHist2 = reg_cnn_model_large.fit(
      data, epochs=15, callbacks=[es,reduce_lr]
    )

reg_cnn_model_large.save("./cnn_large/1", save_format="tf")