import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import geopandas as gpd
import numpy as np

# Set up page and title
st.set_page_config(page_icon=None, layout="wide")

st.markdown("""
    <h1 style='text-align: center; color: black;'>🌪️ Welcome to the Natural Disaster Damage Prediction Tool! 🌪️</h1>
    <h4 style='color: black;'> 🌊 Select a disaster type and enter the relevant details to get an estimate of the property damage.</h2>
    <h4 style='color: black;'>📊 Simply input the attributes of the selected disaster and click the button to predict</h2>
    """, unsafe_allow_html=True)

# Load pre-trained Random Forest models
models = {
    "Tornado": joblib.load("tornado_model.pkl"),
    "Flood": joblib.load("flood_model.pkl"),
    "Lightning": joblib.load("lightning_model.pkl"),
    "High Wind": joblib.load("high_wind_model.pkl"),
    "Wildfire": joblib.load("wildfire_model.pkl"),
    "Tropical Depression": joblib.load("tropical_depression_model.pkl"),
}

# Load datasets
cmap = pd.read_csv("cmap.csv")  # Contains county locations
df = pd.read_csv("merged_data_county.csv")  # Contains GDP per capita and Density for county
geojson_file = 'counties.geojson'
counties_geojson = gpd.read_file(geojson_file)

# Disaster type selection
disaster_type = st.sidebar.selectbox("Select a disaster type:", list(models.keys()), key="disaster_type_selector")
model = models[disaster_type]

# Add definition of each natural disaster for enhanced clarity
disaster_definitions = {
    "Tornado": "A rapidly rotating column of air extending from a thunderstorm to the ground. 🌪️",
    "Flood": "An overflow of water onto normally dry land. 🌊",
    "Lightning": "A sudden electrostatic discharge in the atmosphere.⚡",
    "High Wind": "Sustained strong winds capable of causing damage. 🌬️",
    "Wildfire": "An uncontrolled fire spreading rapidly in natural areas. 🔥 ",
    "Tropical Depression": "A low-pressure weather system with organized thunderstorms. 🌀"
}

# Display definitions
st.sidebar.write(f"**Definition:** {disaster_definitions[disaster_type]}")

# State selection
state_options = sorted(df["State"].dropna().unique())
selected_state = st.sidebar.selectbox("Select a state:", state_options)

# Filter counties based on selected state
state_counties = df[df["State"] == selected_state]["County"].dropna().astype(str).unique()
selected_county = st.sidebar.selectbox("Select a county:", sorted(state_counties))

if len(state_counties) == 0:
    st.error(f"No counties found for the selected state: {selected_state}")
    st.stop()

# Tornado state list
state_list_tor = [
    "State_Arizona", "State_Arkansas", "State_California", "State_Colorado",
    "State_Delaware", "State_District Of Columbia", "State_Florida", "State_Georgia",
    "State_Hawaii", "State_Idaho", "State_Illinois", "State_Indiana", "State_Iowa",
    "State_Kansas", "State_Kentucky", "State_Louisiana", "State_Maine", "State_Maryland",
    "State_Massachusetts", "State_Michigan", "State_Minnesota", "State_Mississippi",
    "State_Missouri", "State_Montana", "State_Nebraska", "State_Nevada",
    "State_New Hampshire", "State_New Jersey", "State_New Mexico", "State_New York",
    "State_North Carolina", "State_North Dakota", "State_Ohio", "State_Oklahoma",
    "State_Oregon", "State_Pennsylvania", "State_Rhode Island", "State_South Carolina",
    "State_South Dakota", "State_Tennessee", "State_Texas", "State_Utah", "State_Vermont",
    "State_Virginia", "State_Washington", "State_West Virginia", "State_Wisconsin", "State_Wyoming"
]

# High Wind state list
state_list_wind =[
    "State_Alaska", "State_Arizona", "State_Arkansas", "State_California", "State_Colorado",
    "State_Delaware", "State_District Of Columbia", "State_Florida", "State_Georgia",
    "State_Hawaii", "State_Idaho", "State_Illinois", "State_Indiana", "State_Iowa",
    "State_Kansas", "State_Kentucky", "State_Louisiana", "State_Maine", "State_Maryland",
    "State_Massachusetts", "State_Michigan", "State_Minnesota", "State_Mississippi",
    "State_Missouri", "State_Montana", "State_Nebraska", "State_Nevada",
    "State_New Hampshire", "State_New Jersey", "State_New Mexico", "State_New York",
    "State_North Carolina", "State_North Dakota", "State_Ohio", "State_Oklahoma",
    "State_Oregon", "State_Pennsylvania", "State_Rhode Island", "State_South Carolina",
    "State_South Dakota", "State_Tennessee", "State_Texas", "State_Utah", "State_Vermont",
    "State_Virginia", "State_Washington", "State_West Virginia", "State_Wisconsin", "State_Wyoming"
]

# Wildfire state list
state_list_fire = [
    "State_Alaska", "State_Arizona", "State_Arkansas", "State_California", "State_Colorado",
    "State_Delaware", "State_Florida", "State_Georgia", "State_Hawaii", "State_Idaho",
    "State_Illinois", "State_Indiana", "State_Iowa", "State_Kansas", "State_Kentucky",
    "State_Louisiana", "State_Maryland", "State_Massachusetts", "State_Michigan",
    "State_Minnesota", "State_Missouri", "State_Montana", "State_Nebraska", 
    "State_Nevada", "State_New Jersey", "State_New Mexico", "State_New York",
    "State_North Carolina", "State_North Dakota", "State_Oklahoma", "State_Oregon",
    "State_Pennsylvania", "State_South Carolina", "State_South Dakota", "State_Tennessee",
    "State_Texas", "State_Utah", "State_Virginia", "State_Washington", 
    "State_West Virginia", "State_Wisconsin", "State_Wyoming"
]

# Lightining state list
state_list_light = [
    "State_Arizona", "State_Arkansas", "State_California", "State_Colorado", "State_Delaware",
    "State_District Of Columbia", "State_Florida", "State_Georgia", "State_Hawaii", "State_Idaho",
    "State_Illinois", "State_Indiana", "State_Iowa", "State_Kansas", "State_Kentucky",
    "State_Louisiana", "State_Maine", "State_Maryland", "State_Massachusetts", "State_Michigan",
    "State_Minnesota", "State_Mississippi", "State_Missouri", "State_Montana", "State_Nebraska",
    "State_Nevada", "State_New Hampshire", "State_New Jersey", "State_New Mexico", "State_New York",
    "State_North Carolina", "State_North Dakota", "State_Ohio", "State_Oklahoma", "State_Oregon",
    "State_Pennsylvania", "State_Rhode Island", "State_South Carolina", "State_South Dakota", "State_Tennessee",
    "State_Texas", "State_Utah", "State_Vermont", "State_Virginia", "State_Washington",
    "State_West Virginia", "State_Wisconsin", "State_Wyoming"
]

# Flood state list
state_list_flood = [
    "State_Alaska", "State_Arizona", "State_Arkansas", "State_California", "State_Colorado", 
    "State_Delaware", "State_District Of Columbia", "State_Florida", "State_Georgia", "State_Hawaii",
    "State_Idaho", "State_Illinois", "State_Indiana", "State_Iowa", "State_Kansas",
    "State_Kentucky", "State_Louisiana", "State_Maine", "State_Maryland", "State_Massachusetts",
    "State_Michigan", "State_Minnesota", "State_Mississippi", "State_Missouri", "State_Montana",
    "State_Nebraska", "State_Nevada", "State_New Hampshire", "State_New Jersey", "State_New Mexico",
    "State_New York", "State_North Carolina", "State_North Dakota", "State_Ohio", "State_Oklahoma",
    "State_Oregon", "State_Pennsylvania", "State_Rhode Island", "State_South Carolina", "State_South Dakota",
    "State_Tennessee", "State_Texas", "State_Utah", "State_Vermont", "State_Virginia",
    "State_Washington", "State_West Virginia", "State_Wisconsin", "State_Wyoming"
]

# Tropical Depression state list
state_list_trop = ["State_Arkansas", "State_Florida", "State_Georgia", "State_Louisiana", "State_Mississippi", "State_South Carolina", "State_Tennessee", "State_Texas"
]

# Initialize all state columns with 0
state_data_tor = {col: 0 for col in state_list_tor}
state_data_wind = {col: 0 for col in state_list_wind}
state_data_fire = {col: 0 for col in state_list_fire}
state_data_light = {col: 0 for col in state_list_light}
state_data_flood = {col: 0 for col in state_list_flood}
state_data_trop = {col: 0 for col in state_list_trop}

# Set the selected state's column to 1
selected_state_cap = selected_state.title()
state_column = f"State_{selected_state_cap}"

# Year selection
year = st.sidebar.number_input("Year:", min_value=2007, max_value=2030, value=2022)

# Check if the selected year is beyond 2022
if year > 2022:
    st.warning("You have selected a year beyond 2022. Choose whether to enter values manually or use predicted data.")
    
    # User choice: Manual input or use data predicte
    choice = st.radio(
        "How do you want to provide GDP per capita and Density?",
        ("Enter manually", "Use predicted data")
    )
    
    if choice == "Enter manually":
        # Manual input of GDP per capita and Density
        gdp_per_capita = st.sidebar.number_input("Enter GDP per capita for the county:", min_value=0.0, step=1000.0)
        density = st.sidebar.number_input("Enter population density for the county (people per sq. km):", min_value=0.0, step=10.0)
    else:
        # Fetch latest data from the dataset
        county_data = df[(df["State"] == selected_state) & (df["County"] == selected_county) & (df["Year"] == 2022)]
        
        if county_data.empty:
            st.error(f"No GDP or Density data found for {selected_county}, {selected_state} in the year 2022.")
            st.stop()
        
        # Extract GDP per capita and Density from the dataset
        gdp_per_capita = county_data["GDP_per_capita"].values[0]
        density = county_data["Density"].values[0]
else:
    # Fetch GDP per capita and Density for the selected county and year
    county_data = df[(df["State"] == selected_state) & 
                     (df["County"] == selected_county) & 
                     (df["Year"] == year)]
    
    if county_data.empty:
        st.error(f"No GDP or Density data found for {selected_county}, {selected_state} in the year {year}.")
        st.stop()
    
    # Extract GDP per capita and Density values
    gdp_per_capita = county_data["GDP_per_capita"].values[0]
    density = county_data["Density"].values[0]


# Sidebar Inputs - Common inputs across all disasters
injuries_direct = st.sidebar.number_input("Number of direct injuries:", min_value=0, value=0)
injuries_indirect = st.sidebar.number_input("Number of indirect injuries:", min_value=0, value=0)
deaths_direct = st.sidebar.number_input("Number of direct deaths:", min_value=0, value=0)
deaths_indirect = st.sidebar.number_input("Number of indirect deaths:", min_value=0, value=0)
duration_hours = st.sidebar.slider("Disaster Duration (hours):", min_value=1, max_value=100, value=10)

# Dynamic Inputs by Disaster Type
if disaster_type == "Tornado":
    distance_tor = st.sidebar.number_input("Distance from the affected area (km):", min_value=0.0, step=0.1)
    tor_length = st.sidebar.number_input("Tornado Length (in km):", min_value=0.0, step=0.1)
    tor_width = st.sidebar.number_input("Tornado Width (in km):", min_value=0.0, step=0.1)
    tor_f_scale = st.sidebar.selectbox("Tornado F-Scale Rating:", 
                                        ["F0", "F1", "F2", "F3", "F4", "EF1", "EF2", "EF3", "EF4", "EF5", "EFU"])
                                        
    if state_column in state_data_tor:
        state_data_tor[state_column] = 1
    else:
        st.error(f"There have not been any recorded tornado in the State of {selected_state} since 2007")
        st.stop()

    # Set one-hot encoding for the F-Scale data
    tor_f_scale_data = {
        "TOR_F_SCALE_EF1": 1 if tor_f_scale == "EF1" else 0,
        "TOR_F_SCALE_EF2": 1 if tor_f_scale == "EF2" else 0,
        "TOR_F_SCALE_EF3": 1 if tor_f_scale == "EF3" else 0,
        "TOR_F_SCALE_EF4": 1 if tor_f_scale == "EF4" else 0,
        "TOR_F_SCALE_EF5": 1 if tor_f_scale == "EF5" else 0,
        "TOR_F_SCALE_EFU": 1 if tor_f_scale == "EFU" else 0,
        "TOR_F_SCALE_F0": 1 if tor_f_scale == "F0" else 0,
        "TOR_F_SCALE_F1": 1 if tor_f_scale == "F1" else 0,
        "TOR_F_SCALE_F2": 1 if tor_f_scale == "F2" else 0,
        "TOR_F_SCALE_F3": 1 if tor_f_scale == "F3" else 0,
        "TOR_F_SCALE_F4": 1 if tor_f_scale == "F4" else 0,  
    }

    # Input Data dictionary structure for Tornado
    input_data = pd.DataFrame({
        "Year": [year],
        "INJURIES_DIRECT": [injuries_direct],
        "INJURIES_INDIRECT": [injuries_indirect],
        "DEATHS_DIRECT": [deaths_direct],
        "DEATHS_INDIRECT": [deaths_indirect],
        "TOR_LENGTH": [tor_length],
        "TOR_WIDTH": [tor_width],
        "DURATION_HOURS": [duration_hours], 
        "Distance_km": [distance_tor],
        "GDP_per_capita": [gdp_per_capita],
        "Density": [density],
        **state_data_tor,  # Ensure state encoding
        **tor_f_scale_data  
    })

if disaster_type == "Flood":
    distance_km_flood = st.sidebar.number_input("Distance from the affected area (km):", min_value=0.0, step=0.1)
    flood_cause = st.sidebar.selectbox("Flood Cause:", 
                                        ["Heavy Rain", "Heavy Rain / Burn Area", "Heavy Rain / Snow Melt", "Heavy Rain / Tropical System", "Ice Jam", "Dam Release"])
    flood_cause_data = {
       "FLOOD_CAUSE_Heavy Rain": 1 if flood_cause == "Heavy Rain" else 0,
       "FLOOD_CAUSE_Heavy Rain / Burn Area": 1 if flood_cause == "Heavy Rain / Burn Area" else 0,
       "FLOOD_CAUSE_Heavy Rain / Snow Melt": 1 if flood_cause == "Heavy Rain / Snow Melt" else 0,
       "FLOOD_CAUSE_Heavy Rain / Tropical System": 1 if flood_cause == "Heavy Rain / Tropical System" else 0,
       "FLOOD_CAUSE_Ice Jam": 1 if flood_cause == "Ice Jam" else 0,
       "FLOOD_CAUSE_Planned Dam Release": 1 if flood_cause == "Dam Release" else 0,     
     }
     
    if state_column in state_data_flood:
       state_data_flood[state_column] = 1
    else:
        st.error(f"There have not been any recorded flood in the State of {selected_state} since 2007")
        st.stop()                       

    input_data = pd.DataFrame({
        **state_data_flood,
        "Year": [year],
        "GDP_per_capita": [gdp_per_capita],
        "Density": [density],
        "DURATION_HOURS": [duration_hours],
        "INJURIES_DIRECT": [injuries_direct],
        "INJURIES_INDIRECT": [injuries_indirect],
        "DEATHS_DIRECT": [deaths_direct],
        "DEATHS_INDIRECT": [deaths_indirect],
        "Distance_km": [distance_km_flood],
        **flood_cause_data
    })

if disaster_type == "High Wind":
    magnitude = st.sidebar.number_input("Magnitude:", min_value=0.0, step=0.1)
    
    if state_column in state_data_wind:
        state_data_wind[state_column] = 1
    else:
        st.error(f"There have not been any recorded high wind in the State of {selected_state} since 2007")
        st.stop()
    
    input_data = pd.DataFrame({
        "Year": [year],
        "INJURIES_DIRECT": [injuries_direct],
        "INJURIES_INDIRECT": [injuries_indirect],
        "DEATHS_DIRECT": [deaths_direct],
        "DEATHS_INDIRECT": [deaths_indirect],
        "MAGNITUDE": [magnitude],
        "DURATION_HOURS": [duration_hours],
        "GDP_per_capita": [gdp_per_capita],
        "Density": [density],
        **state_data_wind,
    })

if disaster_type == "Wildfire":
    magnitude = st.sidebar.number_input("Magnitude:", min_value=0.0, step=0.1)
    
    if state_column in state_data_fire:
        state_data_fire[state_column] = 1
    else:
        st.error(f"There have not been any recorded wildfire in the State of {selected_state} since 2007")
        st.stop()
    
    input_data = pd.DataFrame({
        "Year": [year],
        "INJURIES_DIRECT": [injuries_direct],
        "INJURIES_INDIRECT": [injuries_indirect],
        "DEATHS_DIRECT": [deaths_direct],
        "DEATHS_INDIRECT": [deaths_indirect],
        "DURATION_HOURS": [duration_hours],
        "GDP_per_capita": [gdp_per_capita],
        "Density": [density],
        **state_data_fire,
    })

if disaster_type == "Lightning":
    distance_light = st.sidebar.number_input("Distance from the affected area (km):", min_value=0.0, step=0.1)
    
    if state_column in state_data_light:
        state_data_light[state_column] = 1
    else:
        st.error(f"There have not been any recorded lightning in the State of {selected_state} since 2007")
        st.stop()
    
    input_data = pd.DataFrame({
        "Year": [year],
        "INJURIES_DIRECT": [injuries_direct],
        "INJURIES_INDIRECT": [injuries_indirect],
        "DEATHS_DIRECT": [deaths_direct],
        "DEATHS_INDIRECT": [deaths_indirect],
        "DURATION_HOURS": [duration_hours],
        "Distance_km": [distance_light],
        "GDP_per_capita": [gdp_per_capita],
        "Density": [density],
        **state_data_light,
    })

if disaster_type == "Tropical Depression":
     
    if state_column in state_data_trop:
        state_data_trop[state_column] = 1
    else:
        st.error(f"There have not been any recorded tropical depression in the State of {selected_state} since 2007")
        st.stop()
    
    input_data = pd.DataFrame({
        "Year": [year],
        "INJURIES_DIRECT": [injuries_direct],
        "INJURIES_INDIRECT": [injuries_indirect],
        "DEATHS_DIRECT": [deaths_direct],
        "DEATHS_INDIRECT": [deaths_indirect],
        "DURATION_HOURS": [duration_hours],
        "GDP_per_capita": [gdp_per_capita],
        "Density": [density],
        **state_data_trop,
    })


# Prediction logic
if st.sidebar.button("Predict Property Damage 📊"):
    try:
        with st.spinner("Calculating predictions... Please wait."):
            predicted_damage = model.predict(input_data)
            transformed_damage = np.exp(predicted_damage[0])
        st.subheader("Predicted Damage:")
        st.write(f"${transformed_damage:,.2f}")
    except Exception as e:
        st.error(f"Error making prediction: {e}")
else:
    st.write("Adjust the inputs on the sidebar and click the button to make predictions.")


#County predictions for interactive map
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
import geopandas as gpd
if disaster_type=='Tornado':
    tornado_data = {
        "Year": [year],
        "INJURIES_DIRECT": [injuries_direct],
        "INJURIES_INDIRECT": [injuries_indirect],
        "DEATHS_DIRECT": [deaths_direct],
        "DEATHS_INDIRECT": [deaths_indirect],
        "TOR_LENGTH": [tor_length],
        "TOR_WIDTH": [tor_width],
        "DURATION_HOURS": [duration_hours], 
        "Distance_km": [distance_tor],
        "GDP_per_capita": [gdp_per_capita],
        "Density": [density],
        **state_data_tor,
        **tor_f_scale_data}
    
if disaster_type=='Flood':
    flood_data = {
        **state_data_flood,
        "Year": [year],
        "GDP_per_capita": [gdp_per_capita],
        "Density": [density],
        "DURATION_HOURS": [duration_hours],
        "INJURIES_DIRECT": [injuries_direct],
        "INJURIES_INDIRECT": [injuries_indirect],
        "DEATHS_DIRECT": [deaths_direct],
        "DEATHS_INDIRECT": [deaths_indirect],
        "Distance_km": [distance_km_flood],
        **flood_cause_data
    }
if disaster_type=='High Wind':
    high_wind_data = {
        "Year": [year],
        "INJURIES_DIRECT": [injuries_direct],
        "INJURIES_INDIRECT": [injuries_indirect],
        "DEATHS_DIRECT": [deaths_direct],
        "DEATHS_INDIRECT": [deaths_indirect],
        "MAGNITUDE": [magnitude],
        "DURATION_HOURS": [duration_hours],
        "GDP_per_capita": [gdp_per_capita],
        "Density": [density],
        **state_data_wind,
    }
if disaster_type=='Wildfire':
    wildfire_data = {
        "Year": [year],
        "INJURIES_DIRECT": [injuries_direct],
        "INJURIES_INDIRECT": [injuries_indirect],
        "DEATHS_DIRECT": [deaths_direct],
        "DEATHS_INDIRECT": [deaths_indirect],
        "DURATION_HOURS": [duration_hours],
        "GDP_per_capita": [gdp_per_capita],
        "Density": [density],
        **state_data_fire,
    }
if disaster_type=='Lightning':
    lightning_data ={
        "Year": [year],
        "INJURIES_DIRECT": [injuries_direct],
        "INJURIES_INDIRECT": [injuries_indirect],
        "DEATHS_DIRECT": [deaths_direct],
        "DEATHS_INDIRECT": [deaths_indirect],
        "DURATION_HOURS": [duration_hours],
        "Distance_km": [distance_light],
        "GDP_per_capita": [gdp_per_capita],
        "Density": [density],
        **state_data_light,
    }

if disaster_type == "Tropical Depression":
    tropical_data ={
        "Year": [year],
        "INJURIES_DIRECT": [injuries_direct],
        "INJURIES_INDIRECT": [injuries_indirect],
        "DEATHS_DIRECT": [deaths_direct],
        "DEATHS_INDIRECT": [deaths_indirect],
        "DURATION_HOURS": [duration_hours],
        "GDP_per_capita": [gdp_per_capita],
        "Density": [density],
        **state_data_trop,
    }

import streamlit as st
from streamlit_folium import folium_static
from shapely import wkt
import folium

# Input data based on the selected disaster type
if disaster_type == "Tornado":
    input_data = tornado_data
elif disaster_type == "Flood":
    input_data = flood_data
elif disaster_type == "High Wind":
    input_data = high_wind_data
elif disaster_type == "Wildfire":
    input_data = wildfire_data
elif disaster_type == "Lightning":
    input_data = lightning_data
elif disaster_type == "Tropical Depression":
    input_data = tropical_data

# Now we insert the user inputs in the cmap dataframe, which contains the Geo JSON information and features for each county.
# The GDP per capita and Density must not change based on user inputs as they depend on the county
predicted_damages = []

for key in input_data.keys():
    if key not in cmap.columns and key not in ['Density', 'GDP_per_capita']:
        cmap[key] = None  # Add missing dummy encoded columns

# Loop for cmap rows
for index, row in cmap.iterrows():
    model_input = {}

    for key, value in input_data.items():
        # Skip Density and GDP per capita
        if key in ['Density', 'GDP_per_capita']:
            model_input[key] = row[key] 
        else:
            # Else we use the input data
            model_input[key] = value[0] if isinstance(value, list) else value
            cmap.at[index, key] = model_input[key]  # Update cmap to ensure consistent DataFrame

    # Convert to DataFrame
    model_input_df = pd.DataFrame([model_input])

    prediction = model.predict(model_input_df)

    prediction = np.expm1(prediction)

    predicted_damages.append(prediction[0])  # Store prediction in the predicted damages list

# Add to predicted damage column in cmap
cmap["predicted_damage"] = predicted_damages

# Convert to GeoDataFrame
cmap['geometry'] = cmap['geometry'].apply(wkt.loads)
cmap_gdf = gpd.GeoDataFrame(cmap, geometry='geometry', crs="EPSG:4326")



# Interactive map with spinner
with st.spinner("Loading the map..."):
    # Create interactive map
    m = folium.Map(location=[cmap_gdf.geometry.centroid.y.mean(), cmap_gdf.geometry.centroid.x.mean()], zoom_start=7)

    # Choropleth counties layer
    folium.Choropleth(
        geo_data=cmap_gdf.__geo_interface__,
        name="Predicted Damage",
        data=cmap,
        columns=["NAME", "predicted_damage"],
        key_on="feature.properties.NAME",
        fill_color="Spectral",
        fill_opacity=0.8,
        line_opacity=0.1,
        legend_name="Predicted Damage",
        reset=True
    ).add_to(m)

    # Add tooltips for interactivity
    folium.GeoJson(
        cmap_gdf.__geo_interface__,
        style_function=lambda feature: {
            'fillColor': 'transparent',  # No fill colour
            'color': 'black',
            'weight': 0.1,
            'fillOpacity': 0
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["NAME", "predicted_damage"],
            aliases=["County:", "Predicted Damage:"],
            localize=True
        )
    ).add_to(m)

    # Render the map with folium_static
    folium_static(m, width=1200, height=1000)

st.success("Map successfully loaded!")



