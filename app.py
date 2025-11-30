import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageOps
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Brain Tumor Detection", layout="wide")

# --- SIDEBAR: MODEL SELECTION ---
st.sidebar.header("⚙️ System Status")

# CHANGE THIS FILENAME to whatever you named your trained file
MODEL_PATH = "my_tumor_model.pt" 

try:
    model = YOLO(MODEL_PATH)
    st.sidebar.success("✅ Model Loaded")
    # Removed the classes list display here
    
except Exception as e:
    st.sidebar.error("❌ Model NOT Found!")
    st.sidebar.warning(f"Please find 'best.pt' in your 'runs' folder, copy it here, and rename it to '{MODEL_PATH}'.")
    st.stop() 

conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.01)

# --- HELPER: GET LOCATION ---
def get_position_label(box, img_width, img_height):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    
    if cx < img_width / 3: h_pos = "Left"
    elif cx < 2 * img_width / 3: h_pos = "Center"
    else: h_pos = "Right"
    
    if cy < img_height / 3: v_pos = "Top"
    elif cy < 2 * img_height / 3: v_pos = "Middle"
    else: v_pos = "Bottom"
    
    return f"{v_pos}-{h_pos}"

# --- APP INTERFACE ---
st.title("🧠 Brain Tumor AI Dashboard")
st.markdown("---")

col1, col2 = st.columns(2)

# --- LEFT COLUMN: INPUT ---
with col1:
    st.header("1. Upload Scan")
    uploaded_file = st.file_uploader("Upload MRI Image", type=['jpg', 'png', 'jpeg'])
    
    if uploaded_file:
        try:
            image = Image.open(uploaded_file)
            image = ImageOps.exif_transpose(image)
            
            # Force RGB conversion (Fixes crashes with PNG/Transparency)
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            st.image(image, caption="Original MRI", use_container_width=True)
        except Exception as e:
            st.error(f"Error loading image: {e}")

# --- RIGHT COLUMN: RESULTS ---
with col2:
    st.header("2. AI Analysis Report")
    
    if uploaded_file and st.button("Analyze Image", type="primary"):
        with st.spinner("Analyzing scan pattern..."):
            
            # Save temp file for robust reading
            temp_path = "temp_upload.jpg"
            
            try:
                image.save(temp_path)
                
                # Run inference
                results = model.predict(source=temp_path, conf=conf_threshold, save=False)
                result = results[0]
                
                # Visualize
                res_plotted = result.plot() 
                res_image = res_plotted[:, :, ::-1] # Convert BGR to RGB
                st.image(res_image, caption="AI Detection Result", use_container_width=True)
                
                # Clean up
                if os.path.exists(temp_path):
                    os.remove(temp_path)

                # --- REPORTING ---
                boxes = result.boxes
                if len(boxes) > 0:
                    width, height = image.size
                    for i, box in enumerate(boxes):
                        cls_id = int(box.cls[0])
                        class_name = model.names[cls_id]
                        conf = float(box.conf[0])
                        coords = box.xyxy[0].tolist()
                        
                        if "no tumor" in class_name.lower():
                            st.success(f"✅ **Classification: No Tumor**")
                            st.write("Healthy tissue detected.")
                        else:
                            position = get_position_label(coords, width, height)
                            st.error(f"⚠️ **Detection #{i+1}: {class_name.title()}**")
                            st.write(f"- **Confidence:** {conf:.1%}")
                            st.write(f"- **Location:** {position}")
                else:
                    st.info(f"No patterns detected. (Threshold: {conf_threshold})")
            
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                if os.path.exists(temp_path):
                    os.remove(temp_path)