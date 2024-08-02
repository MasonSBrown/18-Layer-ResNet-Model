import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
import os
import torch.optim.lr_scheduler as lr_scheduler

#Two different datasets are created for training and validation
val_dir = os.path.join('archive', "test") 
train_dir = os.path.join('archive', "train")
#'join' joins the different pieces depending on the slashes appropriate for personal operating system.

#Tranforms needed to be applied to images
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

# Load Datasets, Climbs through folder
# Takes Tranforms and applies them to images
train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms) 
val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)

# Data Loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False) #testing how well model works



# Pretrained ResNet-18 model ******************************************************************

model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1) #Pretrained matrix weights from ImageNet
# model = models.resnet18() #Non Pretrained model

#**********************************************************************************************



num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2) #2 outputs, cat and dog (fc for feature count?)
# Check if MPS is available
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS backend")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using NVIDIA")
else:
    device = torch.device("cpu")
    print("MPS backend not available, using CPU")
model = model.to(device) #setting device to model

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
patience = 5

def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=10, patience=5):
    best_val_loss = float("inf")
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()

        #Training Loss and Accuracy 
        #(Average loss over the entire training dataset)
        #(Accuracy over the entire training dataset)
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects / len(train_loader.dataset)

        print(f'Epoch {epoch}/{num_epochs - 1}')
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data).item()

        #Training Loss and Accuracy 
        #(Average loss over the entire training dataset)
        #(Accuracy over the entire training dataset)
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects / len(val_loader.dataset)


        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')

        if scheduler:
            scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early Stopping Now")
                break

    model.load_state_dict(best_model_wts)
    return model

# Train the model
model = train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=15, patience=patience)
