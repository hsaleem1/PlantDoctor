import os
import requests
import torch

MODEL_PATH = 'best_wheat_model.pth'

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    # Use the direct download URL
    url = "https://drive.usercontent.google.com/download?id=1alRYSZ5CbaYTpRiuR76tRukhYgtrR0ma&export=download&authuser=0"
    
    response = requests.get(url, stream=True)
    
    if response.status_code == 200:
        with open(MODEL_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"✅ Model downloaded! Size: {os.path.getsize(MODEL_PATH) / (1024*1024):.2f} MB")
    else:
        print(f"❌ Download failed! Status code: {response.status_code}")
        # Remove any partial file
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        raise Exception("Model download failed")


import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_PATH = 'best_wheat_model.pth'
CLASS_NAMES = ['BYDV', 'Healthy', 'Septoria']
NUM_CLASSES = 3
IMAGE_SIZE = 224

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu', weights_only=False))
    model.eval()
    return model

model = load_model()

# ============================================================
# TRANSFORMS
# ============================================================
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# ============================================================
# EXPLANATIONS
# ============================================================
def get_explanation(class_name, confidence):
    explanations = {
        'BYDV': {
            'diagnosis': 'Barley Yellow Dwarf Virus (BYDV)',
            'action': 'Scout field within 3 days. Apply aphid control if live aphids present.',
            'urgency': 'HIGH - Can reduce yields by up to 60%'
        },
        'Healthy': {
            'diagnosis': 'Healthy Plant',
            'action': 'Continue routine monitoring. No action needed.',
            'urgency': 'LOW - Maintain good crop management'
        },
        'Septoria': {
            'diagnosis': 'Septoria Leaf Spot',
            'action': 'Apply fungicide within 7-10 days if weather favors spread.',
            'urgency': 'MODERATE - Can reduce yields by 30-50%'
        }
    }
    
    info = explanations.get(class_name, {
        'diagnosis': f'Unknown: {class_name}',
        'action': 'Consult agronomist for verification',
        'urgency': 'UNKNOWN'
    })
    
    level = "High" if confidence > 80 else "Medium" if confidence > 60 else "Low"
    
    return f"""
🌾 **Diagnosis:** {info['diagnosis']}  
📊 **Confidence:** {confidence:.1f}% ({level})  
📋 **Action:** {info['action']}  
⚠️ **Urgency:** {info['urgency']}  
💡 *AI-assisted. Confirm with field scouting.*
"""

# ============================================================
# PREDICTION
# ============================================================
def predict_image(image):
    if image is None:
        return "No image", 0.0, "Please upload an image first."
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    input_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        temperature = 2.0
        probs = torch.nn.functional.softmax(outputs[0] / temperature, dim=0)
        pred_class = torch.argmax(probs).item()
        confidence = probs[pred_class].item() * 100
    
    diagnosis = CLASS_NAMES[pred_class]
    return diagnosis, confidence, get_explanation(diagnosis, confidence)

# ============================================================
# STREAMLIT UI
# ============================================================
st.set_page_config(page_title="PlantDoctor", page_icon="🌾")

st.title("🌾 PlantDoctor - Wheat Disease Detection")
st.write("Upload a wheat leaf image for instant diagnosis and spray recommendations")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with st.spinner("Analyzing..."):
        diagnosis, confidence, explanation = predict_image(image)
    
    st.success(f"🔍 Diagnosis: **{diagnosis}**")
    st.info(f"📊 Confidence: **{confidence:.1f}%**")
    
    st.markdown("---")
    st.markdown("### 📋 Explanation & Spray Recommendations")
    st.markdown(explanation)
    
    st.caption("⚠️ AI-assisted diagnosis. Always confirm with field scouting.")
