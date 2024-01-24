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

class ModifiedFCN(nn.Module):
    def __init__(self, num_classes=1):
        super(ModifiedFCN, self).__init__()
        # Load pre-trained FCN with a specific backbone
        self.fcn = models.segmentation.fcn_resnet101(pretrained=True)
        self.fcn.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=(1, 1))  # Adjusting for the number of output classes

        # Additional layers
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(num_classes, 1)  # Output a single value

    def forward(self, x):
        x = self.fcn(x)['out']
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x

folder_path = 'unet'


scaler = joblib.load(folder_path+'/'+'scaler.save')


# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')
# Step 1: Load the trained model
model_path = folder_path+'/'+'best_model.pth'  # Replace with your model's file path
model = torch.load(model_path)
model = model.to(device)
model.eval()

# Step 2: Create a dataset for test images
test_images_folder = '../data/test'  # Replace with your folder path
test_images = os.listdir(test_images_folder)

# Define transforms (should be the same as used during training, without augmentation)
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_image(image_path):
    with open(image_path, 'rb') as f:
        image = Image.open(f)
        image = image.convert('RGB')
    return image

# Step 3: Run inference
results = []
for img_name in test_images:
    img_path = os.path.join(test_images_folder, img_name)
    image = load_image(img_path)
    image = test_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        predicted_pci = output.cpu().numpy()[0][0]

    results.append({'image_name': img_name, 'PCI': predicted_pci})

# Step 4: Create DataFrame


df = pd.DataFrame(results)
# df['PCI'] = scaler.inverse_transform(df['PCI'])


# Before inverse transforming, reshape df['PCI'] to a 2D array
pci_reshaped = df['PCI'].values.reshape(-1, 1)

# Apply inverse_transform to the reshaped array
inverse_transformed_pci = scaler.inverse_transform(pci_reshaped)

# Assign the result back to the DataFrame
# No need for flatten() as Pandas handles the conversion from 2D array to Series
df['PCI'] = inverse_transformed_pci

df.to_csv(folder_path+'/'+'pci.csv', index=False)

# Step 5: Generate submission file
gen_submit(df)
