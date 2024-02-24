import os
import torch
from torchvision import transforms
from PIL import Image
import pandas as pd
from utils.generate_submission import gen_submit
from sklearn.preprocessing import StandardScaler
import joblib
import torchvision.models as models
import torch.nn as nn
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from utils.sendemail import sendMail
import numpy as np
from torch.utils.data import DataLoader, Dataset

class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the test images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_names = os.listdir(root_dir)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_names[idx])
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        return image, self.image_names[idx]

# Transforms
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Test DataLoader
test_dataset = TestDataset(root_dir='../data/test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')
# Step 1: Load the trained model
folder_path = 'resnet18+MSE+downsampling'
model_path = folder_path+'/'+'best_model.pth'  # Replace with your model's file path
model = torch.load(model_path)
model = model.to(device)
model.eval()

# Step 2: Create a dataset for test images
test_images_folder = '../data/test'  # Replace with your folder path
test_images = os.listdir(test_images_folder)


# def load_image(image_path):
#     with open(image_path, 'rb') as f:
#         image = Image.open(f)
#         image = image.convert('RGB')
#     return image

def load_image(image_path):
    try:
        with open(image_path, 'rb') as f:
            image = Image.open(f)
            image = image.convert('RGB')
        return image
    except Exception as e:
        print(f"Error loading image '{image_path}': {e}")
        return None


# Step 3: Run inference
results = []
for img_name in test_images:
    img_path = os.path.join(test_images_folder, img_name)
    image = load_image(img_path)
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)

        predicted_pci = output.cpu().numpy()[0][0]

    results.append({'image_name': img_name, 'PCI': predicted_pci*100})

df = pd.DataFrame(results)

# df['PCI'] = df['PCI'].astype(int)
df['PCI'] = np.floor(df['PCI']).astype(int)


df.to_csv(folder_path+'/'+'pci.csv', index=False)

# Step 5: Generate submission file
gen_submit(df)