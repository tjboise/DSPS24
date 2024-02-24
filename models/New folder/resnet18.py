import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
import time
import datetime
from utils.sendemail import sendMail
import copy
import os


# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Define a custom dataset
class PavementDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        """
        Args:
            dataframe (pandas.DataFrame): DataFrame containing image paths and PCI.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.dataframe = dataframe
        self.root_dir = '../../data/'+root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.dataframe.iloc[idx, 0])
        image = Image.open(img_name)
        # pci = self.dataframe.iloc[idx,1].apply(lambda x: max(0, min(x, 100)))
        # pci = self.dataframe.iloc[idx, 1]/100 ###############################
        pci = max(0, self.dataframe.iloc[idx, 1]) / 100
        if self.transform:
            image = self.transform(image)
        return image, pci

### downsampling ###
# Load dataset
df = pd.read_csv('../../data/train.csv')  # Adjust path as necessary

# Filter out some images with PCI = 100
df_pci_100 = df[df['pci'] == 100]
df_others = df[df['pci'] != 100]

# Example: Keep 50% of the images with PCI = 100
# You can adjust this percentage based on your needs
sample_size = int(len(df_pci_100) * 0.1)
df_pci_100_sampled = df_pci_100.sample(sample_size)

# Concatenate the sampled df_pci_100 with df_others
df_balanced = pd.concat([df_pci_100_sampled, df_others], ignore_index=True)
print(len(df_balanced))
# Proceed with your data processing
# train_df, val_df = train_test_split(df_balanced, test_size=0.2, random_state=42)
# ---------------------------------------------------------------- #
train_df=df_balanced
val_df=df_balanced

# Transforms
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),

    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Datasets
train_dataset = PavementDataset(dataframe=train_df, root_dir='train', transform=transform)
val_dataset = PavementDataset(dataframe=val_df, root_dir='train', transform=transform)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# Model definition
model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
model.fc = nn.Sequential(
    # nn.Dropout(0.5),  # Add dropout
    nn.Linear(model.fc.in_features, 1),
    nn.Sigmoid()
)
model = model.to(device)

# # Use EfficientNet-B0
# model = models.efficientnet_b0(pretrained=True)  # Ensure you have the correct pretrained argument
# # Replace the classifier layer with your own custom layer
# model.classifier[1] = nn.Sequential(
#     nn.Linear(model.classifier[1].in_features, 1),
#     nn.Sigmoid()
# )
# model = model.to(device)


# Loss and optimizer
# loss_function = nn.MSELoss()

# def combined_loss(output, target, alpha=0.5):
#     # mape_loss = torch.mean(torch.abs((target - output)) / (torch.abs(target)+1e-8))**2
#     mse_loss = torch.mean(torch.abs((target - output))**2)
#     weight_score = torch.mean(1/(torch.abs(target)+1))**2
#     return  weight_score * mse_loss


def custom_weighted_mse_loss(output, target):
    # Calculate weights: higher for smaller target values, lower for larger target values.
    # This example uses an inverse square root scaling. Adjust the scaling function as needed.
    weights = 1 / torch.sqrt(torch.abs(target) + 1)
    squared_diffs = (target - output) ** 2
    weighted_squared_diffs = weights * squared_diffs

    # Calculate the weighted mean of the squared differences.
    loss = torch.mean(weighted_squared_diffs)
    return loss


loss_function = custom_weighted_mse_loss

optimizer = optim.Adam(model.parameters(), lr=0.0005)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

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

        # loss = loss_function(outputs, labels)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_bar.set_postfix(loss=train_loss/len(train_loader))
    print('outputs:',outputs*100)
    print("labels:", labels*100)

    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    print(f'Epoch {epoch + 1}/{your_num_epochs}, Current Learning Rate: {current_lr}')
    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    model.eval()

    val_loss = 0.0
    val_bar = tqdm(val_loader, desc=f'Validation Epoch {epoch+1}/{your_num_epochs}')
    with torch.no_grad():
        for inputs, labels in val_bar:
            inputs, labels = inputs.to(device), labels.to(device)  # Transfer to GPU

            outputs = model(inputs)
            labels = labels.float().unsqueeze(1)
            # loss = loss_function(outputs, labels)
            loss = loss_function(outputs, labels)
            val_loss += loss.item()
            val_bar.set_postfix(loss=val_loss/len(val_loader))

    val_loss /= len(val_loader)
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