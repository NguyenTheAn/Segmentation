

from keras.models import *
from keras.layers import *
from keras import optimizers
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import cv2
import os
import numpy as np
from imutils import paths
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import math

trainAug = dict(
    height_shift_range=0.1, width_shift_range=0.1, zoom_range=0.1)

def adjustData(img,mask, datagen):
    parameters = datagen.get_random_transform((256, 256, 3))
    img = datagen.apply_transform(img, parameters)
    img = img / 255
    mask = datagen.apply_transform(mask, parameters)
    mask = mask /255
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    return (img,mask)


def trainGenerator(batch_size, train_path, image_folder, mask_folder, aug_dict, target_size = (256,256), seed = 1):
  
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = "grayscale",
        target_size = target_size,
        batch_size = batch_size,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = "grayscale",
        target_size = target_size,
        batch_size = batch_size,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img,mask = adjustData(img,mask, image_datagen)
        yield (img,mask)


# Aug = trainGenerator(images, labels, 2,trainAug)
import os
path = os.getcwd()
path = os.path.join(path, "test")


Aug = trainGenerator(2, path, "inputs", "outputs", trainAug)

def castF(x):
    return K.cast(x, K.floatx())

def castB(x):
    return K.cast(x, bool)

def iou_loss_core(true,pred):  #this can be used as a loss if you make it negative
    intersection = true * pred
    notTrue = 1 - true
    union = true + (notTrue * pred)

    return (K.sum(intersection, axis=-1) + K.epsilon()) / (K.sum(union, axis=-1) + K.epsilon())

def IOU(true, pred): #any shape can go - can't be a loss function

    tresholds = [0.5 + (i*.05)  for i in range(10)]

    #flattened images (batch, pixels)
    true = K.batch_flatten(true)
    pred = K.batch_flatten(pred)
    pred = castF(K.greater(pred, 0.5))

    #total white pixels - (batch,)
    trueSum = K.sum(true, axis=-1)
    predSum = K.sum(pred, axis=-1)

    #has mask or not per image - (batch,)
    true1 = castF(K.greater(trueSum, 1))    
    pred1 = castF(K.greater(predSum, 1))

    #to get images that have mask in both true and pred
    truePositiveMask = castB(true1 * pred1)

    #separating only the possible true positives to check iou
    testTrue = tf.boolean_mask(true, truePositiveMask)
    testPred = tf.boolean_mask(pred, truePositiveMask)

    #getting iou and threshold comparisons
    iou = iou_loss_core(testTrue,testPred) 
    truePositives = [castF(K.greater(iou, tres)) for tres in tresholds]

    #mean of thressholds for true positives and total sum
    truePositives = K.mean(K.stack(truePositives, axis=-1), axis=-1)
    truePositives = K.sum(truePositives)

    #to get images that don't have mask in both true and pred
    trueNegatives = (1-true1) * (1 - pred1) # = 1 -true1 - pred1 + true1*pred1
    trueNegatives = K.sum(trueNegatives) 

    return (truePositives + trueNegatives) / castF(K.shape(true)[0])

#layer 1
inputs = Input((256, 256, 3))
conv1 = Conv2D(32, 3, activation='relu', padding='same',
               kernel_initializer='he_normal')(inputs)
conv1 = Conv2D(32, 3, activation='relu', padding='same',
               kernel_initializer='he_normal')(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

conv2 = Conv2D(64, 3, activation='relu', padding='same',
               kernel_initializer='he_normal')(pool1)
conv2 = Conv2D(64, 3, activation='relu', padding='same',
               kernel_initializer='he_normal')(conv2)

drop2 = Dropout(0.5)(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(drop2)


# -------------------------------------------------------------
conv3 = Conv2D(128, 3, activation='relu', padding='same',
               kernel_initializer='he_normal')(pool2)
conv3 = Conv2D(128, 3, activation='relu', padding='same',
               kernel_initializer='he_normal')(conv3)
drop3 = Dropout(0.5)(conv3)
# -------------------------------------------------------------
up4 = Conv2D(64, 2, activation='relu', padding='same',
             kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop3))
merge4 = concatenate([drop2, up4], axis=3)
conv4 = Conv2D(64, 3, activation='relu', padding='same',
               kernel_initializer='he_normal')(merge4)
conv4 = Conv2D(64, 3, activation='relu', padding='same',
               kernel_initializer='he_normal')(conv4)

up5 = Conv2D(32, 2, activation='relu', padding='same',
             kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv4))
merge5 = concatenate([conv1, up5], axis=3)
conv5 = Conv2D(32, 3, activation='relu', padding='same',
               kernel_initializer='he_normal')(merge5)
conv5 = Conv2D(32, 3, activation='relu', padding='same',
               kernel_initializer='he_normal')(conv5)

conv6 = Conv2D(1, 1, activation='sigmoid', name="output_tensor")(conv5)

model = Model(inputs, conv6)
    
def jaccard_distance(y_true, y_pred, smooth=100):
    """ Calculates mean of Jaccard distance as a loss function """
    intersection = tf.reduce_sum(y_true * y_pred, axis=(1,2))
    sum_ = tf.reduce_sum(y_true + y_pred, axis=(1,2))
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    jd =  (1 - jac) * smooth
    return tf.reduce_mean(jd)
  
optimizer = optimizers.SGD()
model.compile(optimizer="adam", loss=["binary_crossentropy"], metrics=[IOU])

# model.summary()

model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
H = model.fit_generator(Aug, steps_per_epoch=50, 
                        callbacks=[model_checkpoint], epochs=100)

N = 20
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
# plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
# plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")