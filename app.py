# Add these to your existing imports
import matplotlib.pyplot as plt
import io
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import os
from huggingface_hub import hf_hub_download

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
    # Download from Hugging Face
    try:
        model_path = hf_hub_download(
            repo_id="Muhammad-Hammad-Saleem/PlantDoctor-model",
            filename="best_wheat_model.pth"
        )
    except Exception as e:
        st.error(f"Failed to download model: {e}")
        st.stop()
    
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=False))
    model.eval()
    return model

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
# GRAD-CAM FUNCTION
# ============================================================
def generate_gradcam(image):
    """Generate Grad-CAM heatmap for the input image"""
    img_np = np.array(image)
    input_tensor = transform(image).unsqueeze(0)
    
    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        pred_class = torch.argmax(probs).item()
        confidence = probs[pred_class].item() * 100
    
    # Generate Grad-CAM heatmap
    cam = GradCAM(model=model, target_layers=[model.layer4[-1]])
    targets = [ClassifierOutputTarget(pred_class)]
    heatmap = cam(input_tensor=input_tensor, targets=targets)
    heatmap = heatmap[0, :]
    
    # Resize heatmap to match image
    heatmap_resized = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
    
    # Overlay heatmap on image
    img_display = img_np.astype(np.float32) / 255.0
    visualization = show_cam_on_image(img_display, heatmap_resized, use_rgb=True)
    
    # Convert to RGB for display
    visualization_rgb = cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB)
    
    return {
        'class': CLASS_NAMES[pred_class],
        'confidence': confidence,
        'heatmap': visualization_rgb,
        'original': img_np
    }

# ============================================================
# PLAIN ENGLISH EXPLANATION
# ============================================================
def generate_explanation(class_name, confidence):
    """Convert technical prediction to farmer-friendly language"""
    
    explanations = {
        'BYDV': {
            'diagnosis': 'Barley Yellow Dwarf Virus (BYDV)',
            'evidence': 'Yellowing and stunting patterns detected',
            'action': 'Scout field within 3 days. Apply aphid control if live aphids present.',
            'urgency': 'HIGH - Can reduce yields by up to 60%'
        },
        'Healthy': {
            'diagnosis': 'Healthy Plant',
            'evidence': 'No disease symptoms detected',
            'action': 'Continue routine monitoring. No action needed.',
            'urgency': 'LOW - Maintain good crop management'
        },
        'Septoria': {
            'diagnosis': 'Septoria Leaf Spot',
            'evidence': 'Brown lesions with yellow halos detected',
            'action': 'Apply fungicide within 7-10 days if weather favors spread.',
            'urgency': 'MODERATE - Can reduce yields by 30-50%'
        }
    }
    
    info = explanations.get(class_name, {
        'diagnosis': f'Unknown: {class_name}',
        'evidence': 'Consult agronomist for verification',
        'action': 'Send this image to your agricultural advisor',
        'urgency': 'UNKNOWN'
    })
    
    confidence_level = "High" if confidence > 80 else "Medium" if confidence > 60 else "Low"
    
    explanation = f"""
🌾 DIAGNOSIS: {info['diagnosis']}
📊 Confidence: {confidence:.1f}% ({confidence_level})
🔍 Evidence: {info['evidence']}
📋 Action: {info['action']}
⚠️ Urgency: {info['urgency']}
💡 Note: This is an AI-assisted diagnosis. Always confirm with field scouting.
"""
    

# ============================================================
# PREDICTION (with Grad-CAM)
# ============================================================
def predict_image(image):
    if image is None:
        return "No image", 0.0, "Please upload an image first.", None
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Get Grad-CAM result
    result = generate_gradcam(image)
    
    return result['class'], result['confidence'], get_explanation(result['class'], result['confidence']), result['heatmap']

# ============================================================
# LOAD MODEL (Cached)
# ============================================================
model = load_model()

# ============================================================
# STREAMLIT UI
# ============================================================
st.set_page_config(page_title="PlantDoctor", page_icon="🌾")

st.title("🌾 PlantDoctor - Wheat Disease Detection")
st.write("Upload a wheat leaf image for instant diagnosis and spray recommendations")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    
    with st.spinner("Analyzing..."):
        diagnosis, confidence, explanation, heatmap = predict_image(image)
    
    # Display original and heatmap side by side
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with col2:
        st.image(heatmap, caption=f"Grad-CAM Heatmap: {diagnosis} ({confidence:.1f}%)", use_container_width=True)
    
    st.success(f"🔍 Diagnosis: **{diagnosis}**")
    st.info(f"📊 Confidence: **{confidence:.1f}%**")
    
    st.markdown("---")
    st.markdown("### 📋 Explanation & Spray Recommendations")
    st.markdown(explanation)
    
    st.caption("⚠️ AI-assisted diagnosis. Always confirm with field scouting.")
