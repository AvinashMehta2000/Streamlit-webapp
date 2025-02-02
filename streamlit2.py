import streamlit as st
import requests
import torch
from io import BytesIO
from PIL import Image, ImageDraw
import numpy as np
from ultralytics import YOLO

#Set Streamlit layout to wide st.set_page_config(layout="wide")

# Constants
ZOOM_LEVEL = 17
IMAGE_SIZE = 640  # px

def get_bounding_box(lat, lon, zoom=17, size=640):
    """
    Convert center (lat, lon) to bounding box.
    Uses approximate scale for web mercator (EPSG:3857) at zoom level 17.
    
    Args:
        lat (float): Center latitude
        lon (float): Center longitude
        zoom (int): Zoom level (default: 17)
        size (int): Image size in pixels (default: 640)
    
    Returns:
        tuple: (xmin, ymin, xmax, ymax) bounding box in WGS84
    """
    # Approximate scale factor for zoom level 16 (based on Web Mercator tile size)
    scale = 156543.03 / (2 ** zoom)  # meters per pixel

    # Convert pixel size to degrees (assuming ~111km per degree at equator)
    offset = (size * scale) / 111000  # Convert meters to degrees

    xmin = lon - offset
    xmax = lon + offset
    ymin = lat - offset
    ymax = lat + offset

    return (xmin, ymin, xmax, ymax)

def get_esri_basemap_image(lat, lon, size=640, zoom=17):
    """
    Fetch a 640x640 basemap image from Esri using a center lat/lon.
    
    Args:
        lat (float): Center latitude
        lon (float): Center longitude
        size (int): Image size in pixels
        zoom (int): Zoom level
    
    Returns:
        PIL.Image: The fetched basemap image.
    """
    bbox = get_bounding_box(lat, lon, zoom, size)
    url = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export"
    params = {
        "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
        "bboxSR": 4326,
        "size": f"{size},{size}",
        "imageSR": 4326,
        "format": "png",
        "f": "image"
    }
    response = requests.get(url, params=params)
    image = Image.open(BytesIO(response.content))
    return image

@st.cache_resource
def load_model():
    return YOLO("yolo11lv3.pt")  # for lacally saved weight

model = load_model()

# --- Streamlit UI ---
st.title("🛰️ YOLO Object Detection on Esri Basemap 🗺️")
st.subheader("Fine-tuned for Brick Kiln Detection")

st.write(
    "This web app demonstrates a **YOLO-based object detection model fine-tuned to detect brick kilns** on a satellite basemap. "
    "To test the model's performance at different locations, simply enter the **center latitude and longitude**, and the app "
    "will fetch a **640px satellite image at zoom level 17** from Esri's basemap."
)

st.warning(" **Note:** This web-app showcases the model's performance visually, allowing you to test it at any location. ")

# External link centered
st.markdown(
    """
    <div style="text-align: center;">
        <a href="https://brickkiln-locations.streamlit.app/" target="_blank">
            🔗 Go to Brick Kiln Explorer: Model Detected BK across IGP States
        </a>
    </div>
    """, 
    unsafe_allow_html=True
)
st.write("")
st.sidebar.header("🌍 Enter Center Coordinates")
lat = st.sidebar.number_input("Latitude", value=28.745802, format="%.6f")
lon = st.sidebar.number_input("Longitude", value=77.417503, format="%.6f")

# Run Detection Button
if st.sidebar.button("Run Detection"):
    with st.spinner("Fetching ESRI basemap image..."):
        basemap_image = get_esri_basemap_image(lat, lon)
    
    st.image(basemap_image, caption="Esri Basemap", use_container_width=True)

    # Convert to NumPy array for YOLO
    image_np = np.array(basemap_image.convert("RGB"))

    with st.spinner("Running object detection..."):
        results = model(image_np, conf=0.5)

    # Annotate detections
    annotated_image = basemap_image.convert("RGB")
    draw = ImageDraw.Draw(annotated_image)

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = box[:4]
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

    st.image(annotated_image, caption="Detection Results", use_container_width=True)
    
# Sidebar: Model Details
st.sidebar.subheader("📌 Model Details")
st.sidebar.markdown(
    """
    <style>
        .model-details p {
            margin-bottom: 2px;  /* Reduces spacing */
            font-size: 16px;  /* Adjusts text size */           
        }
    </style>
    <div class="model-details">
        <p><b>Model:</b> YOLO11-l</p>
        <p><b>Pretrained:</b> Yes (COCO)</p>
        <p><b>Train images:</b> >700</p>
        <p><b>Train Instances:</b> >2500</p>
        <p><b>Val Instances:</b> 358</p>
        <p><b>Epochs:</b> 500</p>
        <p><b>Trained on NVIDIA RTX5000 Ada</b></p>
        <p></p>
        <p></p>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.subheader("📌 Accuracy Metrics")
col1, col2, col3 = st.sidebar.columns(3)
col1.metric(label="🎯 Precision %", value="95.3")
col2.metric(label="🔍 Recall %", value="97.2") 
col3.metric(label="📊 mAP50 %", value="98.2")

# Author details in a styled box at the bottom of the sidebar
st.sidebar.markdown(
    """
    <style>
    .author-box {
        position: relative;
        bottom: 20px;
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin-top: 50px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #2e86de;
    }
    .author-box h3 {
        color: #2e86de;
        margin-top: 0;
    }
    </style>
    
    <div class="author-box">
        <h3>📌 Author Details</h3>
        <p><strong>Avinash Mehta</strong></p>
        <div style="display: flex; gap: 15px; margin-top: 10px;">
            <a href="https://www.linkedin.com/in/avinash-mehta-115624240" target="_blank" style="text-decoration: none;">
                <img src="https://img.icons8.com/color/48/000000/linkedin.png" width="28" height="28">
                <span style="color: #2e86de; vertical-align: super;">LinkedIn</span>
            </a>
            <a href="https://github.com/AvinashMehta2000" target="_blank" style="text-decoration: none;">
                <img src="https://img.icons8.com/color/48/000000/github.png" width="28" height="28">
                <span style="color: #2e86de; vertical-align: super;">GitHub</span>
            </a>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)