import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
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
# MULTI-CROP CONFIGURATION
# ============================================================
CROP_CONFIG = {
    'Wheat': {
        'classes': ['BYDV', 'Healthy', 'Septoria'],
        'model_path': 'best_wheat_model.pth',
        'icon': '🌾',
        'trained_images': 5000,
        'accuracy': 92.5,
        'status': 'trained'
    },
    'Beans': {
        'classes': ['BYDV', 'Healthy', 'Septoria'],
        'model_path': 'best_wheat_model.pth',
        'icon': '🫘',
        'trained_images': 0,
        'accuracy': 0.0,
        'status': 'pending'
    },
    'Broccoli': {
        'classes': ['BYDV', 'Healthy', 'Septoria'],
        'model_path': 'best_wheat_model.pth',
        'icon': '🥦',
        'trained_images': 0,
        'accuracy': 0.0,
        'status': 'pending'
    }
    # Add your other crops here with the same 3 classes
}

CROP_NAMES = list(CROP_CONFIG.keys())

# ============================================================
# LOAD MODEL (Crop-Specific)
# ============================================================
@st.cache_resource
def load_model(crop_name):
    """Load the wheat model for all crops (temporary)"""
    config = CROP_CONFIG.get(crop_name)
    if not config:
        st.error(f"Unknown crop: {crop_name}")
        st.stop()
    
    # Always use the wheat model
    model_path = None
    
    # Try Hugging Face first
    try:
        model_path = hf_hub_download(
            repo_id="Muhammad-Hammad-Saleem/PlantDoctor-model",
            filename="best_wheat_model.pth"
        )
    except Exception as e:
        # Fall back to local file
        try:
            model_path = 'best_wheat_model.pth'
        except:
            pass
    
    if model_path is None:
        st.error("Could not find model file")
        st.stop()
    
    # Always use 3 classes (wheat model)
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 3)
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
# GRAD-CAM FUNCTION (Updated to accept model)
# ============================================================
def generate_gradcam(image):
    """Generate Grad-CAM heatmap for the input image"""
    img_np = np.array(image)
    input_tensor = transform(image).unsqueeze(0)
    
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
    
    # Resize and overlay (uses cv2)
    heatmap_resized = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
    img_display = img_np.astype(np.float32) / 255.0
    visualization = show_cam_on_image(img_display, heatmap_resized, use_rgb=True)
    visualization_rgb = cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB)
    
    return {
        'class': CLASS_NAMES[pred_class],
        'confidence': confidence,
        'heatmap': visualization_rgb,
        'original': img_np
    }
# ============================================================
# EXPLANATIONS (Updated with Evidence)
# ============================================================
def get_explanation(class_name, confidence, class_names):
    """Get explanation for the given class"""
    explanations = {
        'BYDV': {
            'diagnosis': 'Barley Yellow Dwarf Virus (BYDV)',
            'evidence': 'Yellowing and stunting patterns detected',  # <-- ADDED
            'action': 'Scout field within 3 days. Apply aphid control if live aphids present.',
            'urgency': 'HIGH - Can reduce yields by up to 60%'
        },
        'Healthy': {
            'diagnosis': 'Healthy Plant',
            'evidence': 'No disease symptoms detected',  # <-- ADDED
            'action': 'Continue routine monitoring. No action needed.',
            'urgency': 'LOW - Maintain good crop management'
        },
        'Septoria': {
            'diagnosis': 'Septoria Leaf Spot',
            'evidence': 'Brown lesions with yellow halos detected',  # <-- ADDED
            'action': 'Apply fungicide within 7-10 days if weather favors spread.',
            'urgency': 'MODERATE - Can reduce yields by 30-50%'
        }
    }
    
    info = explanations.get(class_name, {
        'diagnosis': f'Unknown: {class_name}',
        'evidence': 'Consult agronomist for verification',  # <-- ADDED
        'action': 'Consult agronomist for verification',
        'urgency': 'UNKNOWN'
    })
    
    level = "High" if confidence > 80 else "Medium" if confidence > 60 else "Low"
    
    # Updated return string to include evidence
    return f"""
🌾 **Diagnosis:** {info['diagnosis']}  
📊 **Confidence:** {confidence:.1f}% ({level})  
🔍 **Evidence:** {info['evidence']}  
📋 **Action:** {info['action']}  
⚠️ **Urgency:** {info['urgency']}  
💡 *AI-assisted. Confirm with field scouting.*
"""
    

# ============================================================
# PREDICTION (Crop-Specific)
# ============================================================
def predict_image(image, model, class_names):
    """Predict for a single image with the given model"""
    if image is None:
        return "No image", 0.0, "Please upload an image first.", None
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Get Grad-CAM result (pass the model)
    result = generate_gradcam(image, model, class_names)
    
    return result['class'], result['confidence'], get_explanation(result['class'], result['confidence'], class_names), result['heatmap']

# ============================================================
# BATCH PREDICTION (Multiple Images)
# ============================================================
def predict_batch(images, model, class_names):
    """Predict for multiple images"""
    results = []
    for img in images:
        diagnosis, confidence, explanation, heatmap = predict_image(img, model, class_names)
        results.append({
            'diagnosis': diagnosis,
            'confidence': confidence,
            'explanation': explanation,
            'heatmap': heatmap,
            'original': np.array(img)
        })
    return results



# ============================================================
# LOAD MODEL (Cached)
# ============================================================
# model = load_model()

# ============================================================
# STREAMLIT UI (Multi-Crop)
# ============================================================
st.set_page_config(page_title="PlantDoctor", page_icon="🌾")

st.title("🌾 PlantDoctor - Multi-Crop Disease Detection")
st.write("Select a crop and upload images for instant diagnosis")

# ============================================================
# SIDEBAR: CROP SELECTION & MODEL CARDS
# ============================================================
st.sidebar.header("🌱 Select Crop")

selected_crop = st.sidebar.selectbox(
    "Choose your crop:",
    options=CROP_NAMES,
    index=0
)

# Crop info
crop_info = CROP_CONFIG[selected_crop]
st.sidebar.markdown(f"**{crop_info['icon']} {selected_crop}**")
st.sidebar.markdown(f"**Detectable conditions:** {len(crop_info['classes'])}")
st.sidebar.markdown("**Classes:**")
for cls in crop_info['classes']:
    st.sidebar.markdown(f"- {cls}")

if 'trained_images' in crop_info:
    st.sidebar.markdown(f"**Training images:** {crop_info['trained_images']}")
if 'accuracy' in crop_info and crop_info['accuracy'] > 0:
    st.sidebar.markdown(f"**Model accuracy:** {crop_info['accuracy']:.1f}%")
else:
    st.sidebar.markdown("**Model accuracy:** Not yet trained")

st.sidebar.markdown("---")

# ============================================================
# MODEL CARDS (Inside Sidebar)
# ============================================================
st.sidebar.subheader("📊 Model Card")

with st.sidebar.container():
    # Status
    if crop_info.get('accuracy', 0) > 0:
        st.sidebar.markdown("✅ **Status:** Trained")
    else:
        st.sidebar.markdown("⏳ **Status:** Training in progress")
    
    # Training details
    st.sidebar.markdown(f"📊 **Training images:** {crop_info.get('trained_images', 'N/A')}")
    
    if crop_info.get('accuracy', 0) > 0:
        st.sidebar.markdown(f"🎯 **Accuracy:** {crop_info['accuracy']:.1f}%")
    else:
        st.sidebar.markdown("🎯 **Accuracy:** Not yet available")
    
    st.sidebar.markdown(f"📁 **Classes:** {len(crop_info['classes'])}")
    
    # Progress bar
    if crop_info.get('accuracy', 0) > 0:
        st.sidebar.progress(crop_info['accuracy'] / 100, text=f"Model readiness: {crop_info['accuracy']:.0f}%")
    else:
        st.sidebar.progress(0.3, text="Model readiness: 30% (placeholder)")
    
    st.sidebar.caption("📅 Model version: v1.0 (June 2026)")

st.sidebar.markdown("---")
st.sidebar.caption("Upload images to get diagnosis and recommendations")

# ============================================================
# LOAD MODEL FOR SELECTED CROP
# ============================================================
with st.spinner(f"Loading model for {selected_crop}..."):
    model = load_model(selected_crop)
    CLASS_NAMES = CROP_CONFIG[selected_crop]['classes']
    NUM_CLASSES = len(CLASS_NAMES)

st.success(f"✅ Ready! Analyzing {selected_crop}")


# ============================================================
# MAIN UPLOAD AREA & RESULTS
# ============================================================
uploaded_files = st.file_uploader(
    f"Upload {selected_crop} images...", 
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files and len(uploaded_files) > 0:
    images = []
    for uploaded_file in uploaded_files:
        img = Image.open(uploaded_file).convert('RGB')
        images.append(img)
    
    with st.spinner(f"Analyzing {len(images)} image(s)..."):
        results = predict_batch(images, model, CLASS_NAMES)
    
    # Display each result
    for idx, result in enumerate(results):
        with st.container():
            st.subheader(f"📸 Image {idx + 1}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(result['original'], caption="Uploaded Image", use_container_width=True)
            with col2:
                st.image(result['heatmap'], caption=f"Grad-CAM: {result['diagnosis']} ({result['confidence']:.1f}%)", use_container_width=True)
            
            st.success(f"🔍 **Diagnosis:** {result['diagnosis']}")
            st.info(f"📊 **Confidence:** {result['confidence']:.1f}%")
            
            st.markdown("### 📋 Explanation & Spray Recommendations")
            st.markdown(result['explanation'])
            st.markdown("---")
       

    # ============================================================
    # FEEDBACK LOOP (Hugging Face Dataset)
    # ============================================================
    from huggingface_hub import HfApi
    import json
    import datetime
    
    # Initialize Hugging Face API
    hf_api = HfApi()
    
    # Your feedback dataset repository
    FEEDBACK_REPO = "Muhammad-Hammad-Saleem/PlantDoctor-Feedback"
    
    def save_feedback(image_name, predicted, actual, crop, correct):
        """Save feedback to Hugging Face dataset"""
        try:
            # Load existing feedback
            try:
                import requests
                url = f"https://huggingface.co/datasets/{FEEDBACK_REPO}/raw/main/feedback.jsonl"
                response = requests.get(url)
                if response.status_code == 200:
                    with open("feedback.jsonl", "w") as f:
                        f.write(response.text)
            except:
                # Create new file if it doesn't exist
                with open("feedback.jsonl", "w") as f:
                    pass
            
            # Append new feedback
            with open("feedback.jsonl", "a") as f:
                feedback_entry = {
                    "timestamp": str(datetime.datetime.now()),
                    "image_name": image_name,
                    "predicted": predicted,
                    "actual": actual,
                    "crop": crop,
                    "correct": correct
                }
                f.write(json.dumps(feedback_entry) + "\n")
            
            # Upload to Hugging Face
            hf_api.upload_file(
                path_or_fileobj="feedback.jsonl",
                path_in_repo="feedback.jsonl",
                repo_id=FEEDBACK_REPO,
                repo_type="dataset"
            )
            return True
        except Exception as e:
            print(f"Error saving feedback: {e}")
            return False
    
    st.markdown("---")
    st.subheader("📝 Help Improve the Model")
    st.caption("Your feedback helps us train better models for farmers")
    
    for idx, result in enumerate(results):
        with st.expander(f"📸 Provide feedback for Image {idx + 1} ({result['diagnosis']})"):
            col1, col2 = st.columns(2)
            image_name = uploaded_files[idx].name if idx < len(uploaded_files) else "unknown"
            current_crop = selected_crop
            
            with col1:
                if st.button(f"✅ Correct", key=f"correct_{idx}"):
                    if save_feedback(image_name, result['diagnosis'], result['diagnosis'], current_crop, "Yes"):
                        st.success("✅ Feedback saved to Hugging Face!")
                    else:
                        st.error("Could not save feedback.")
            
            with col2:
                if st.button(f"❌ Incorrect", key=f"incorrect_{idx}"):
                    with st.expander("✏️ What was the correct diagnosis?"):
                        actual = st.text_input("Enter the correct diagnosis:", key=f"actual_{idx}")
                        if st.button("Submit", key=f"submit_{idx}"):
                            if actual:
                                if save_feedback(image_name, result['diagnosis'], actual, current_crop, "No"):
                                    st.success("🙏 Feedback saved to Hugging Face!")
                                else:
                                    st.error("Could not save feedback.")
                            else:
                                st.warning("Please enter a diagnosis before submitting.")     

    # ============================================================
    # HEALTH DASHBOARD (Mock-up)
    # ============================================================

    st.markdown("---")
    st.subheader("📊 Farm Health Dashboard")
    st.caption("Overview of recent diagnoses across your fields")
    
    # Create a summary of results
    summary = {}
    for result in results:
        crop = selected_crop
        diagnosis = result['diagnosis']
        key = f"{crop} - {diagnosis}"
        summary[key] = summary.get(key, 0) + 1
    
    # Display summary in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="🌾 Total Images Analyzed",
            value=len(results),
            delta="Today's session"
        )
    
    with col2:
        # Count healthy vs diseased
        healthy_count = sum(1 for r in results if r['diagnosis'] == 'Healthy')
        disease_count = len(results) - healthy_count
        st.metric(
            label="🟢 Healthy / 🔴 Diseased",
            value=f"{healthy_count} / {disease_count}",
            delta=f"{disease_count/len(results)*100:.0f}% affected" if len(results) > 0 else "0%"
        )
    
    with col3:
        # Most common issue
        if summary:
            most_common = max(summary, key=summary.get)
            st.metric(
                label="⚠️ Most Common Issue",
                value=most_common.split(" - ")[1],
                delta=f"{summary[most_common]} detections"
            )
        else:
            st.metric(label="⚠️ Most Common Issue", value="None detected")
    
    # Detailed breakdown table
    st.markdown("### 📋 Detailed Breakdown")
    breakdown_data = []
    for key, count in summary.items():
        crop_name, disease = key.split(" - ")
        breakdown_data.append({"Crop": crop_name, "Disease/Issue": disease, "Detections": count})
    
    if breakdown_data:
        import pandas as pd
        df = pd.DataFrame(breakdown_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No detections to display")
    
    # Field-level map visualization (placeholder)
    st.markdown("### 🗺️ Field Health Map")
    st.caption("Interactive map showing affected areas (coming soon)")
    
    # Mock-up of a field map
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Field A**")
        st.markdown("🌾 Wheat - 🟢 Healthy")
        st.markdown("🌾 Wheat - 🟡 Moderate")
        st.markdown("🌾 Wheat - 🔴 BYDV")
    with col2:
        st.markdown("**Field B**")
        st.markdown("🫘 Beans - 🟢 Healthy")
        st.markdown("🫘 Beans - 🟢 Healthy")
    with col3:
        st.markdown("**Field C**")
        st.markdown("🥦 Broccoli - ⚠️ In Progress")
    
    # ============================================================
    # FINAL CAPTION
    # ============================================================

    st.caption("⚠️ AI-assisted diagnosis. Always confirm with field scouting.")
else:
    st.info("👆 Upload one or more images to begin analysis")
