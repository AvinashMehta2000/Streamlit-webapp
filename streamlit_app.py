import streamlit as st
import pandas as pd
import folium
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from streamlit.components.v1 import html

# Load CSV data directly (replace with your CSV file path or URL)
FILE_ID = "1mfsFGPEPASokwJ1nkIY4NmY54cgXl1Md"
CSV_DATA = f"https://drive.google.com/uc?export=download&id={FILE_ID}"


def get_city_coords(city_name):
    """Get coordinates for a city using OpenStreetMap Nominatim"""
    geolocator = Nominatim(user_agent="brickkiln_mapper")
    location = geolocator.geocode(city_name)
    if location:
        return (location.latitude, location.longitude)
    return None

def main():
    st.title("🗺️ Brick Kiln Location Explorer")
    
    st.markdown(
        """
        Explore model-detected brick kilns across the **Indo-Gangetic Plain (IGP)**. 
        You can view brick kilns in any city within a specified radius.
        
        ### 🔍 How to Explore:
        1. **Navigate directly** using the cursor.
        2. **Search for a city** by entering its name in the sidebar.
        3. **Define the search area** by setting a radius (in kilometers).
        """,
        unsafe_allow_html=True
    )
    # Load CSV data directly
    df = pd.read_csv(CSV_DATA)

    # Create map centered on the mean of brickkiln coordinates
    m = folium.Map(location=[df.latitude.mean(), df.longitude.mean()], 
                  zoom_start=5,
                  tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                  attr='Esri World Imagery')

    # Add brick kiln locations to the map with color differentiation
    # Add brick kiln locations with conditional coloring
    for _, kiln in df.iterrows():
        kiln_coords = (kiln["latitude"], kiln["longitude"])
        
        # Check if the city search is active and kiln is within the radius
        if search_city:
            if city_coords:  # Check if city_coords are valid (not None)
                distance = geodesic(city_coords, kiln_coords).km
                if distance <= search_radius:
                    color = "red"  # 🔴 Mark kilns inside the search radius
                else:
                    color = "blue"  # 🔵 Mark kilns outside the search radius
            else:
                st.error(f"City '{search_city}' not found!")
                color = "blue"  # Default color if city coordinates are not found
        else:
            color = "blue"  # Default color if no city is entered
    
        # Plot the kiln with the assigned color
        folium.CircleMarker(
            location=[kiln["latitude"], kiln["longitude"]],
            radius=3,  # Increased size for better visibility
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=1,
            popup=f"<b>{kiln.get('name', 'Brick Kiln')}</b>"
        ).add_to(m)


    # Search UI (additional feature)
    st.sidebar.header("🏙️ City Search")
    search_city = st.sidebar.text_input("Enter city name (e.g., 'Varanasi'):")
    search_radius = st.sidebar.slider("Search radius (km)", 1, 100, 20)

    if search_city:
        # Get city coordinates
        city_coords = get_city_coords(search_city)
        
        if not city_coords:
            st.error(f"City '{search_city}' not found!")
            return

        # Add city marker with popup
        folium.Marker(
            city_coords,
            popup=f"<b>{search_city}</b>",
            icon=folium.Icon(color='green', icon='city')
        ).add_to(m)

        # Add search radius circle (no fill, simple line)
        folium.Circle(
            location=city_coords,
            radius=search_radius*1000,  # Convert km to meters
            color='yellow',
            fill=False,  # No fill color
            weight=2  # Line thickness
        ).add_to(m)

        # Find brickkilns within radius
        nearby_brickkilns = []
        for _, brickkiln in df.iterrows():
            brickkiln_coords = (brickkiln['latitude'], brickkiln['longitude'])
            distance = geodesic(city_coords, brickkiln_coords).km
            if distance <= search_radius:
                brickkiln['distance'] = round(distance, 2)
                nearby_brickkilns.append(brickkiln)

        if nearby_brickkilns:
            st.success(f"Found {len(nearby_brickkilns)} brickkilns within {search_radius} km of {search_city}")
        else:
            st.warning(f"No brickkilns found within {search_radius} km of {search_city}")

    # Display map
    html(m._repr_html_(), width=1000, height=800)

if __name__ == "__main__":
    main()

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
