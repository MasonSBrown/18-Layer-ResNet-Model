import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import pandas as pd
import zipfile
import os
import matplotlib.pyplot as plt

#Two different datasets are created for training and validation
val_dir = os.path.join('archive', "val")
train_dir = os.path.join('archive', "train")

#Parameters
batch_size = 32
img_height = 224
img_width = 224

#Loads images from specified directory and creates a 'tf.data.Dataset' object
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    seed=123, #random seed for shuffling and transformations
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir,
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

print("Tensorflow is intalled. Version:", tf.__version__)

#Residual block is defined
def residual_block(parameters):
    return

#ResNet18 model is built using the residual blocks
def build_resnet18(input_shape, num_classes):
    return

