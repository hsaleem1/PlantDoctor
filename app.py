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
# MULTI-CROP CONFIGURATION
# ============================================================
CROP_CONFIG = {
    'Wheat': {
        'classes': ['BYDV', 'Healthy', 'Septoria'],
        'model_path': 'best_wheat_model.pth',  # Your existing model
        'icon': '🌾'
    },
    'Beans': {
        'classes': ['BYDV', 'Healthy', 'Septoria'],  # Adjust as needed
        'model_path': 'best_barley_model.pth',  # You'll need to train this
        'icon': '🌾'
    },
    'Broccoli': {
        'classes': ['Healthy', 'Blight', 'Leaf Spot'],  # Adjust as needed
        'model_path': 'best_tomato_model.pth',
        'icon': '🍅'
    },
    # Add your 7 crops here
}

# Get list of crop names for dropdown
CROP_NAMES = list(CROP_CONFIG.keys())

# ============================================================
# LOAD MODEL (Crop-Specific)
# ============================================================
@st.cache_resource
def load_model(crop_name):
    """Load the model for the selected crop"""
    config = CROP_CONFIG.get(crop_name)
    if not config:
        st.error(f"Unknown crop: {crop_name}")
        st.stop()
    
    model_path = config['model_path']
    num_classes = len(config['classes'])
    
    try:
        # Try to download from Hugging Face first (if available)
        try:
            from huggingface_hub import hf_hub_download
            model_path = hf_hub_download(
                repo_id=f"Muhammad-Hammad-Saleem/{crop_name.lower()}-model",
                filename="best_model.pth"
            )
        except:
            # Fall back to local file (if running locally)
            pass
        
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=False))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Failed to load model for {crop_name}: {e}")
        st.stop()

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
def get_explanation(class_name, confidence, class_names):
    """Get explanation for the given class"""
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
# GRAD-CAM FUNCTION (Updated to accept model)
# ============================================================
def generate_gradcam(image, model, class_names):
    """Generate Grad-CAM heatmap with the given model"""
    img_np = np.array(image)
    input_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        pred_class = torch.argmax(probs).item()
        confidence = probs[pred_class].item() * 100
    
    cam = GradCAM(model=model, target_layers=[model.layer4[-1]])
    targets = [ClassifierOutputTarget(pred_class)]
    heatmap = cam(input_tensor=input_tensor, targets=targets)
    heatmap = heatmap[0, :]
    
    # Resize and overlay (your existing code)
    # ... keep your existing heatmap generation code
    
    return {
        'class': class_names[pred_class],
        'confidence': confidence,
        'heatmap': visualization_rgb,
        'original': img_np
    }

# ============================================================
# STREAMLIT UI (Multi-Crop)
# ============================================================
st.set_page_config(page_title="PlantDoctor", page_icon="🌾")

st.title("🌾 PlantDoctor - Multi-Crop Disease Detection")
st.write("Select a crop and upload images for instant diagnosis")

# Sidebar for crop selection
st.sidebar.header("🌱 Select Crop")
selected_crop = st.sidebar.selectbox(
    "Choose your crop:",
    options=CROP_NAMES,
    index=0
)

# Display crop info in sidebar
crop_info = CROP_CONFIG[selected_crop]
st.sidebar.markdown(f"**{crop_info['icon']} {selected_crop}**")
st.sidebar.markdown(f"**Detectable conditions:** {len(crop_info['classes'])}")
st.sidebar.markdown("**Classes:**")
for cls in crop_info['classes']:
    st.sidebar.markdown(f"- {cls}")

st.sidebar.markdown("---")
st.sidebar.caption("Upload images to get diagnosis and recommendations")

# Load the selected model
with st.spinner(f"Loading model for {selected_crop}..."):
    model = load_model(selected_crop)
    CLASS_NAMES = CROP_CONFIG[selected_crop]['classes']
    NUM_CLASSES = len(CLASS_NAMES)

st.success(f"✅ Ready! Analyzing {selected_crop}")

# Main upload area
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
    
    st.caption("⚠️ AI-assisted diagnosis. Always confirm with field scouting.")
else:
    st.info("👆 Upload one or more images to begin analysis")
