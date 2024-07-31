import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import zipfile
import os
import matplotlib.pyplot as plt

#Two different datasets are created for training and validation
val_dir = os.path.join('archive', "test")
train_dir = os.path.join('archive', "train")

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)), #Makes all images 224 by 224 pixels
    transforms.RandomHorizontalFlip(), #randomly flips and helps generalize the model better
    transforms.ToTensor(), #convers image from PIL format to Pytorch tensor (necessary for input)
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    #Normalizes images with "ImageNet" statistics, which ResNet was trained on.
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


