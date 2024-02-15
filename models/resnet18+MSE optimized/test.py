import json
import os
import torch
from torchvision import transforms
from PIL import Image
import pandas as pd
# from utils.generate_submission import gen_submit
import torchvision.models as models
import torch.nn as nn
from tqdm import tqdm

folder_path = '.'



# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Step 1: Load the trained model
model_path = folder_path+'/'+'best_model.pth'  # Replace with your model's file path
model = torch.load(model_path)
model = model.to(device)
model.eval()

# Step 2: Create a dataset for test images
test_images_folder = '../../data/test' 
# test_images = os.listdir(test_images_folder)

# Define transforms (should be the same as used during training, without augmentation)
test_transform = transforms.Compose([
    transforms.ToTensor(),
])

def load_image(image_path):
    with open(image_path, 'rb') as f:
        image = Image.open(f)
        image = image.convert('RGB')
    return image

data = pd.read_csv('../../data/test.csv')
target = data['PCI'].values
test_images = data['image_name'].values

# Step 3: Run inference
results = []
for img_name in tqdm(test_images):
    img_path = os.path.join(test_images_folder, img_name)
    image = load_image(img_path)
    image = test_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        predicted_pci = output.cpu().numpy()[0][0]

    results.append({'image_name': img_name, 'PCI': predicted_pci})


def mape_loss(output, target):
    return torch.mean(torch.abs((target - output) / target)) * 100


def gen_submit(df):
    out_json = []
    for _, results in df.iterrows():
        if results['PCI'] < 30:
            continue
        out_json.append({results['image_name']: results['PCI']})
    with open('submission.json', 'w') as f:
        json.dump(out_json, f)


df = pd.DataFrame(results)
df["PCI"] *= 100  # denormalize the PCI value

print(f'MAPE loss: {mape_loss(torch.tensor(df["PCI"].values), torch.tensor(target))}')
# for testing, write a dataframe with the expected and actialy PCI values
testing = pd.DataFrame({'Target': target, 'Predicted': df['PCI'].values})
testing.to_csv('testing.csv', index=False)


gen_submit(df)
