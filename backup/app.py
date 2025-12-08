import os
import json
from flask import Flask, request, render_template, redirect
from werkzeug.utils import secure_filename
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

app = Flask(__name__)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load ResNet50 model architecture and weights
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 102)  # 102 classes
model.load_state_dict(torch.load('model/state_dict.pth', map_location=device))
model.to(device)
model.eval()

# Load category to name mapping from cat_to_name.json
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# Create a mapping from ImageFolder indices to original indices
# This assumes the folder structure is the same as during training
# We simulate the ImageFolder index assignment (alphabetical order of folder names 1 to 102)
# Folder names are "1", "2", ..., "102", so ImageFolder assigns indices 0 to 101
# (e.g., "1" -> 0, "10" -> 1, "100" -> 2, ..., "99" -> 101)
folder_names = [str(i) for i in range(1, 103)]  # "1" to "102"
folder_names.sort()  # Sort alphabetically as ImageFolder does
idx_to_class = {i: int(folder_names[i]) for i in range(len(folder_names))}  # e.g., {0: 1, 1: 10, 2: 100, ..., 101: 99}

# Image preprocessing pipeline (same as in your notebook)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
])

def predict_flower(image_path):
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        predicted_idx = predicted.item()  # This is the ImageFolder index (0 to 101)
        original_idx = idx_to_class[predicted_idx]  # Map to original index (1 to 102)
        return cat_to_name[str(original_idx)]  # Map original index to flower name

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return redirect('/')
    file = request.files['image']
    if file.filename == '':
        return redirect('/')
    
    filename = secure_filename(file.filename)
    upload_dir = 'static/uploads'
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, filename)
    file.save(file_path)

    # Use the predict_flower function to get the prediction
    flower_name = predict_flower(file_path)

    return render_template('index.html', prediction=flower_name, filename=filename)

if __name__ == '__main__':
    app.run(debug=True)