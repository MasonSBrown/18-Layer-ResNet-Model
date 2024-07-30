import tensorflow as tf
# from tensorflow.keras import layers, models
import numpy as np
import os
import matplotlib.pyplot as plt

try:
    import tensorflow as tf
    print("TensorFlow is installed. Version:", tf.__version__)
except ImportError:
    print("TensorFlow is not installed.")
    
#Residual block is defined
def residual_block(parameters):
    return

#ResNet18 model is built using the residual blocks
def build_resnet18(input_shape, num_classes):
    return

#Two different datasets are created for training and validation
train_dataset = 0
validation_dataset = 0