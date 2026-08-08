import os
import requests
import torch
from pathlib import Path

MODEL_PATH = Path('best_wheat_model.pth')

def download_model():
    """Download the model file from Google Drive with validation."""
    # If the model already exists, try to verify it
    if MODEL_PATH.exists():
        try:
            torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
            print(f"✅ Model file '{MODEL_PATH}' already exists and is valid.")
            return True
        except Exception as e:
            print(f"⚠️ Existing model file is corrupted ({e}). Re-downloading...")
            MODEL_PATH.unlink()  # Delete the corrupt file

    # Your file ID from Google Drive
    file_id = "1alRYSZ5CbaYTpRiuR76tRukhYgtrR0ma"
    
    # Use the "uc" endpoint which handles large files better
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    print("📥 Downloading model (44.8 MB)...")
    try:
        # Start the session
        session = requests.Session()
        response = session.get(url, stream=True)
        
        # Check if we got a confirmation page (for large files)
        if "confirm" in response.text:
            # Extract the confirmation token
            import re
            confirm_token = re.search(r'confirm=([^&]+)', response.text)
            if confirm_token:
                confirm_url = f"https://drive.google.com/uc?export=download&confirm={confirm_token.group(1)}&id={file_id}"
                response = session.get(confirm_url, stream=True)
        
        # Proceed with download
        if response.status_code == 200:
            # Check if it's actually a file or HTML
            content_type = response.headers.get('content-type', '')
            if 'text/html' in content_type:
                print("❌ Google Drive returned an HTML page instead of the file.")
                print("Please check your link or make the file publicly accessible.")
                return False
            
            # Download the file
            with open(MODEL_PATH, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        # Print progress every 5 MB
                        if downloaded % (5 * 1024 * 1024) < 8192:
                            print(f"   Downloaded: {downloaded / (1024*1024):.1f} MB")
            
            file_size = MODEL_PATH.stat().st_size
            print(f"✅ Downloaded: {file_size / (1024*1024):.2f} MB")
            
            # Verify the file
            try:
                test_data = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
                print("✅ Model file is valid!")
                return True
            except Exception as e:
                print(f"❌ Downloaded file is corrupted: {e}")
                MODEL_PATH.unlink()  # Delete the corrupt file
                return False
        else:
            print(f"❌ Download failed with status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Download error: {e}")
        return False

# ============================================================
# Load the model (with caching)
# ============================================================
@st.cache_resource
def load_model():
    # Try to download if needed
    if not download_model():
        st.error("Failed to download model. Please check your internet connection and Google Drive link.")
        st.stop()
    
    # Load the model
    try:
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu', weights_only=False))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import os
import requests

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
