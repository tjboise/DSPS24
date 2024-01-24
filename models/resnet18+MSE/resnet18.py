import torch
import pandas as pd
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
import time
import datetime
from utils.sendemail import sendMail
import copy
from sklearn.preprocessing import StandardScaler
import joblib


# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')


def load_image(image_path):
    # Open the image file
    with open(image_path, 'rb') as f:
        image = Image.open(f)
        image = image.convert('RGB')  # Convert to RGB if not already
    return image



class CustomDataset(Dataset):
    def __init__(self, datasets, image_paths, pci_values, transform=None):
        # Convert to list if they are Pandas Series
        self.image_paths = ['../../data/' + datasets + '/'+ path for path in image_paths.tolist()]
        self.pci_values = pci_values.tolist()
        self.transform = transform

        # rest of the class remains the same

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = load_image(self.image_paths[idx])  # Implement this function
        pci = self.pci_values[idx]
        if self.transform:
            image = self.transform(image)
        return image, pci

# Load your data
train_data = pd.read_csv('../../data/train.csv')
test_data = pd.read_csv('../../data/test.csv')

scaler = StandardScaler()

# Fit the scaler to the PCI values and transform
train_data['PCI'] = scaler.fit_transform(train_data[['PCI']])

# Transform the test data using the same scaler
test_data['PCI'] = scaler.transform(test_data[['PCI']])


joblib.dump(scaler, 'scaler.save')


# print(test_data['PCI'])
# train_data, val_data = train_test_split(data, test_size=0.2)

train_data = train_data.reset_index(drop=True)
test_data = test_data.reset_index(drop=True)

# Define transforms


# Define transforms with Data Augmentation for training
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create datasets
train_dataset = CustomDataset('train', train_data['image_name'], train_data['PCI'], train_transform)
test_dataset = CustomDataset('test',test_data['image_name'], test_data['PCI'], test_transform)


# Data loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


# Model
# model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
# model.fc = torch.nn.Linear(model.fc.in_features, 1)  # Modify for regression
# model = model.to(device)

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Sequential(
    nn.Dropout(0.5),  # Add dropout
    nn.Linear(model.fc.in_features, 1)
)
model = model.to(device)


# Loss Function

# Choose the loss function
# loss_function = mape_loss  # Use MAPE loss or nn.MSELoss()

# def mape_loss(output, target):
#     return torch.mean(torch.abs((target - output) / target)) * 100


# If using MAPE loss, add a small epsilon to the denominator to avoid division by zero
# def mape_loss(output, target, epsilon=1e-8):
#     return torch.mean(torch.abs((target - output) / (target + epsilon))) * 100

# Use MAPE loss with an epsilon
# loss_function = lambda output, target: mape_loss(output, target, epsilon=1e-8)
loss_function = nn.MSELoss()
# loss_function = mape_loss
# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Training Loop
# Training Loop with Early Stopping
train_losses = []
val_losses = []
your_num_epochs = 300
# best_val_loss = np.inf
best_model_params = copy.deepcopy(model.state_dict())  # Add this line
best_val_loss = np.inf
patience = 10
trigger_times = 0

best_model = None


for epoch in range(your_num_epochs):
    model.train()
    train_loss = 0.0
    train_bar = tqdm(train_loader, desc=f'Training Epoch {epoch+1}/{your_num_epochs}')
    for inputs, labels in train_bar:
        inputs, labels = inputs.to(device), labels.to(device)  # Transfer to GPU

        optimizer.zero_grad()
        outputs = model(inputs)
        labels = labels.float().unsqueeze(1)  # Ensure correct shape and type
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_bar.set_postfix(loss=train_loss/len(train_loader))

    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    print(f'Epoch {epoch + 1}/{your_num_epochs}, Current Learning Rate: {current_lr}')
    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    model.eval()

    val_loss = 0.0
    val_bar = tqdm(test_loader, desc=f'Validation Epoch {epoch+1}/{your_num_epochs}')
    with torch.no_grad():
        for inputs, labels in val_bar:
            inputs, labels = inputs.to(device), labels.to(device)  # Transfer to GPU

            outputs = model(inputs)
            labels = labels.float().unsqueeze(1)
            loss = loss_function(outputs, labels)
            val_loss += loss.item()
            val_bar.set_postfix(loss=val_loss/len(test_loader))

    val_loss /= len(test_loader)
    val_losses.append(val_loss)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = copy.deepcopy(model)
        # best_model_params = copy.deepcopy(model.state_dict())  # Update the best model
        trigger_times = 0
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print('Early stopping!')
            break


losses = pd.DataFrame({'Train-loss':train_losses, 'Val-loss':val_losses})

# Save DataFrame to CSV
losses.to_csv('loss-epoch.csv', index=False)


# Save the model after training
model_save_path = 'last_model.pth'
best_model_save_path = 'best_model.pth'

torch.save(model, model_save_path)
# Save the best model
if best_model is not None:
    torch.save(best_model, best_model_save_path)
else:
    print("No best model found.")

print(f'Models saved: Last model to {model_save_path}, Best model to {best_model_save_path}')

# torch.save(best_model_params,best_model_save_path )
# print(f'Model saved to {model_save_path}')

message = 'Hi Tianjie, \n This is message to inform you that your training in the lab is already done!'
subject = 'TJ, The network training is finsished at %s! ' % (datetime.datetime.fromtimestamp(time.time()))
sendMail(message, subject)