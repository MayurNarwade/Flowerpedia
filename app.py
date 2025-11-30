import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from PIL import Image
from torchvision import models, transforms

app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 102)
model.load_state_dict(torch.load('model/state_dict.pth', map_location=device))
model.to(device)
model.eval()

# Load class name mappings
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# Load flower metadata from JSON
with open('flower_metadata.json', 'r') as f:
    flower_metadata = json.load(f)

# Map model outputs to actual flower labels
folder_names = [str(i) for i in range(1, 103)]
folder_names.sort()
idx_to_class = {i: int(folder_names[i]) for i in range(len(folder_names))}

# Prepare gallery data with flower names instead of indices
gallery_data = {}
for idx, info in flower_metadata.items():
    flower_name = cat_to_name.get(idx, "Unknown Flower")  # Map index to flower name
    gallery_data[flower_name] = info  # Use flower name as key

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Confidence threshold for OOD detection
CONFIDENCE_THRESHOLD = 0.3

def predict_flower(image_path):
    try:
        # Load and preprocess the image
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Run inference
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            max_prob, predicted_idx = torch.max(probabilities, 1)
        
        # Check confidence
        if max_prob.item() < CONFIDENCE_THRESHOLD:
            return "Invalid image or flower not found", None, max_prob.item()
        
        # Map predicted index to flower name
        original_idx = str(idx_to_class[predicted_idx.item()])
        flower_name = cat_to_name.get(original_idx, "Unknown Flower")
        return flower_name, original_idx, max_prob.item()
    
    except Exception as e:
        return f"Error processing image: {str(e)}", None, 0.0

@app.route('/')
def index():
    return render_template('index.html', gallery_data=gallery_data)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('index.html', error="No image uploaded.", gallery_data=gallery_data)
    file = request.files['image']
    if file.filename == '':
        return render_template('index.html', error="No image selected.", gallery_data=gallery_data)
    if not file.mimetype.startswith('image/'):
        return render_template('index.html', error="Invalid file type.", gallery_data=gallery_data)

    filename = secure_filename(file.filename)
    upload_dir = 'static/uploads'
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, filename)
    file.save(file_path)

    try:
        # Predict
        prediction, original_idx, confidence = predict_flower(file_path)
        
        if prediction == "Invalid image or flower not found" or original_idx is None:
            return render_template('index.html',
                                 error=prediction,
                                 filename=filename,
                                 gallery_data=gallery_data)
        
        # Retrieve metadata
        flower_data = flower_metadata.get(original_idx, None)
        if flower_data:
            flower_info = {
                "common_name": flower_data.get("name", prediction),
                "description": flower_data.get("description", "No description available."),
                "image_url": flower_data.get("image_url")
            }
        else:
            flower_info = {
                "common_name": prediction,
                "description": "No metadata available.",
                "image_url": None
            }
        
        # Include confidence in prediction message
        prediction_message = f"{prediction} (Confidence: {confidence:.2f})"
        
        return render_template('index.html',
                             prediction=prediction_message,
                             filename=filename,
                             flower_info=flower_info,
                             gallery_data=gallery_data)

    except Exception as e:
        return render_template('index.html', error=f"Error: {str(e)}", gallery_data=gallery_data)

if __name__ == '__main__':
    app.run(debug=True)