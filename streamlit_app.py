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
    st.title("🗺️ Brick Kiln Locations Explorer")
    st.markdown("Explore brickkilns near any city")

    # Load CSV data directly
    df = pd.read_csv(CSV_DATA)

    # Create map centered on the mean of brickkiln coordinates
    m = folium.Map(location=[df.latitude.mean(), df.longitude.mean()], 
                  zoom_start=5,
                  tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                  attr='Esri World Imagery')

    # Add all brickkiln points directly to the map (blue dots)
    for _, brickkiln in df.iterrows():
        folium.CircleMarker(
            location=[brickkiln['latitude'], brickkiln['longitude']],
            radius=2,  # Small blue dot
            color='blue',
            fill=True,
            fill_color='blue',
            fill_opacity=1,
            popup=f"<b>{brickkiln.get('name', 'brickkiln')}</b>"
        ).add_to(m)

    # Search UI (additional feature)
    st.sidebar.header("City Search")
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
    html(m._repr_html_(), width=1200, height=600)


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
            margin-top: 300px;
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

    # # Author details in the sidebar
    # st.sidebar.markdown("")  
    # st.sidebar.markdown("")  
    # st.sidebar.markdown("")  
    # st.sidebar.markdown("")  
    # st.sidebar.markdown("")  
    # st.sidebar.markdown("")  
    # st.sidebar.markdown("")  
    # st.sidebar.markdown("")  
    # st.sidebar.markdown("")  
    # st.sidebar.header("Author Details")
    # st.sidebar.markdown("**Avinash Mehta**")
    
    # # LinkedIn and GitHub links with logos
    # st.sidebar.markdown(
    #     """
    #     <div style="display: flex; align-items: center; gap: 10px;">
    #         <a href="https://www.linkedin.com/in/avinash-mehta-115624240" target="_blank">
    #             <img src="https://img.icons8.com/color/48/000000/linkedin.png" width="24" height="24">
    #         </a>
    #         <a href="https://github.com/AvinashMehta2000" target="_blank">
    #             <img src="https://img.icons8.com/color/48/000000/github.png" width="24" height="24">
    #         </a>
    #     </div>
    #     """,
    #     unsafe_allow_html=True
    # )

if __name__ == "__main__":
    main()