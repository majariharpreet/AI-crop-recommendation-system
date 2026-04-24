import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import os
import requests
import folium
from streamlit_folium import st_folium
import time

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Crop Recommender", 
    page_icon="🌾", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- IMPROVED CSS FOR MAXIMUM CONTRAST & FULL UI CONTROL ---
st.markdown("""
    <style>
    /* =========================================
       1. NUKE ALL NATIVE STREAMLIT UI CHROME
       ========================================= */
    
    /* Hide Sidebar completely */
    [data-testid="stSidebar"] { display: none !important; }
    
    /* Hide Sidebar Toggle Arrow */
    [data-testid="collapsedControl"] { display: none !important; }
    
    /* Hide the top Header bar */
    [data-testid="stHeader"] { display: none !important; }
    header { visibility: hidden !important; height: 0px !important; }
    
    /* Hide the floating Toolbar (The stubborn black box on the right!) */
    [data-testid="stToolbar"] { display: none !important; visibility: hidden !important; }
    
    /* Hide the Main Menu Hamburger */
    #MainMenu { display: none !important; visibility: hidden !important; }
    
    /* Hide Footer */
    footer { display: none !important; visibility: hidden !important; }
    [data-testid="stFooter"] { display: none !important; }

    /* =========================================
       2. MAIN APP STYLING
       ========================================= */
       
    /* Background Image with Dark Overlay */
    .stApp {
        background: linear-gradient(rgba(0, 0, 0, 0.6), rgba(0, 0, 0, 0.6)), 
                    url("https://images.unsplash.com/photo-1586771107445-d3ca888129ff?q=80&w=2072&auto=format&fit=crop") no-repeat center center fixed;
        background-size: cover;
    }
    
    /* Main Content Container - highly opaque white card */
    .block-container {
        background-color: rgba(255, 255, 255, 0.95) !important;
        border-radius: 15px;
        padding: 2rem 3rem !important;
        margin-top: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }

    /* FORCE ALL STANDARD TEXT TO BE PITCH BLACK */
    html, body, p, span, label, li {
        color: #000000 !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Force Headers to be Dark Green */
    h1, h2, h3, h4, h5, h6 {
        color: #1B5E20 !important;
        font-weight: 800 !important;
    }
    
    /* Protect Alert Boxes (Success/Error/Info) */
    .stAlert p, .stAlert span, .stAlert div {
        color: inherit !important;
    }
    
    /* Metric Cards Styling */
    [data-testid="stMetric"] {
        background-color: #E8F5E9 !important;
        padding: 15px !important;
        border-radius: 10px !important;
        border: 1px solid #A5D6A7 !important;
    }
    div[data-testid="stMetricValue"] {
        color: #0D47A1 !important; 
        font-size: 32px !important;
        font-weight: 900 !important;
    }
    div[data-testid="stMetricLabel"] p {
        color: #2E7D32 !important; 
        font-weight: 800 !important;
        font-size: 16px !important;
    }

    /* Button Styling */
    .stButton>button {
        background-color: #2E7D32 !important;
        color: white !important;
        border-radius: 8px;
        padding: 10px 24px;
        font-size: 18px;
        font-weight: bold;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    .stButton>button:hover {
        background-color: #1B5E20 !important;
        color: white !important;
    }
    
    /* Fix Tabs Visibility */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #f5f5f5;
        border-radius: 8px 8px 0 0;
    }
    .stTabs [data-baseweb="tab"] p {
        color: #2E7D32 !important;
        font-weight: 700 !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- MACHINE LEARNING MODEL SETUP ---
@st.cache_resource
def load_and_train_model():
    file_path = 'Crop_recommendation.csv'
    if not os.path.exists(file_path):
        return None, None
    
    data = pd.read_csv(file_path)
    X = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = data['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
    svm_model.fit(X_train_scaled, y_train)
    
    return svm_model, scaler

# --- API FUNCTIONS ---
def fetch_weather(lat, lon):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m"
    try:
        response = requests.get(url).json()
        temp = response['current']['temperature_2m']
        humidity = response['current']['relative_humidity_2m']
        return temp, humidity
    except:
        return None, None

def fetch_location_and_rainfall(lat, lon):
    url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}&zoom=10&addressdetails=1"
    headers = {'User-Agent': 'CropRecommenderApp/1.0'} 
    
    try:
        response = requests.get(url, headers=headers).json()
        address = response.get('address', {})
        state = address.get('state', 'Unknown')
        
        state_rainfall_db = {
            "Punjab": 65.0, "Haryana": 60.0, "Himachal Pradesh": 150.0, "Rajasthan": 40.0,
            "Gujarat": 80.0, "Maharashtra": 120.0, "Karnataka": 110.0, "Kerala": 250.0,
            "Tamil Nadu": 95.0, "Andhra Pradesh": 90.0, "Telangana": 90.0, "Madhya Pradesh": 105.0,
            "Uttar Pradesh": 90.0, "Bihar": 120.0, "West Bengal": 160.0, "Odisha": 150.0,
            "Assam": 250.0, "Meghalaya": 300.0, "Uttarakhand": 140.0, "Chhattisgarh": 130.0,
            "Jharkhand": 120.0
        }
        
        rainfall = state_rainfall_db.get(state, 100.0)
        return state, rainfall
    except:
        return "Unknown Region", 100.0

# --- APP HEADER ---
st.title("🌾 AI Crop Recommendation System")
st.markdown("##### *Optimize your agricultural yield using Support Vector Machines and Live Satellite Data.*")
st.markdown("---")

# Load model
model, scaler = load_and_train_model()

if model is None:
    st.error("⚠️ Dataset missing! Please make sure 'Crop_recommendation.csv' is in the exact same folder as this Python file.")
    st.stop()

# --- TABS LAYOUT ---
tab1, tab2 = st.tabs(["🌍 Map Selection (Automated)", "🎛️ Manual Data Entry"])

# --- TAB 1: MAP BASED RECOMMENDATION ---
with tab1:
    st.markdown("### 📍 Step 1: Select Farm Location")
    st.write("Click anywhere on the interactive map. The system will automatically fetch live climate data and historical seasonal rainfall for that region.")
    
    # Map input area 
    m = folium.Map(location=[31.06, 76.24], zoom_start=6)
    map_data = st_folium(m, width=1200, height=450, returned_objects=["last_clicked"])
    
    st.markdown("---")
    
    # Results area
    if map_data and map_data.get("last_clicked"):
        lat = map_data["last_clicked"]["lat"]
        lon = map_data["last_clicked"]["lng"]
        
        with st.spinner("📡 Triangulating location, fetching satellite weather, and analyzing historical rainfall..."):
            time.sleep(1.5) 
            temp, humidity = fetch_weather(lat, lon)
            state, est_rainfall = fetch_location_and_rainfall(lat, lon)
        
        if temp and humidity:
            st.success(f"✅ Data retrieved successfully for coordinates in **{state}**")
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🌡️ Live Temperature", f"{temp} °C")
            with col2:
                st.metric("💧 Live Humidity", f"{humidity} %")
            with col3:
                st.metric("🌧️ Seasonal Rainfall", f"{est_rainfall} mm")
                
            st.markdown("### 🌱 Intelligent Crop Suggestions")
            st.write("Based on regional rainfall and live weather, here are the most viable crops depending on specific soil conditions:")
            
            soil_profiles = {
                "High Nitrogen": [120, 40, 40, 6.5],
                "High Phosphorus": [40, 120, 40, 6.5],
                "High Potassium": [40, 40, 120, 6.5],
                "Balanced/Neutral Soil": [60, 60, 60, 7.0],
                "Low Nutrients": [20, 20, 20, 6.0]
            }
            
            predictions_made = {}
            
            for profile_name, nutrients in soil_profiles.items():
                n, p, k, ph = nutrients
                features = np.array([[n, p, k, temp, humidity, ph, est_rainfall]])
                scaled_features = scaler.transform(features)
                pred = model.predict(scaled_features)[0]
                
                if pred not in predictions_made:
                    predictions_made[pred] = []
                predictions_made[pred].append(profile_name)
            
            for crop, profiles in predictions_made.items():
                profile_text = " or ".join(profiles)
                st.info(f"🌾 **{crop.upper()}** — Ideal if soil has **{profile_text}**")
        else:
            st.error("❌ Could not fetch weather data. Please check your internet connection.")

# --- TAB 2: MANUAL ENTRY ---
with tab2:
    st.markdown("### 🎛️ Input Environmental Parameters")
    
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Soil Nutrients**")
            N = st.slider("Nitrogen (N)", 0, 140, 90)
            P = st.slider("Phosphorus (P)", 0, 145, 42)
            K = st.slider("Potassium (K)", 0, 205, 43)
            ph = st.slider("Soil pH", 3.5, 9.5, 6.5)

        with col2:
            st.markdown("**Climate Conditions**")
            temp_manual = st.slider("Temperature (°C)", 5.0, 45.0, 20.8)
            humidity_manual = st.slider("Humidity (%)", 15.0, 100.0, 82.0)
            rainfall_manual = st.slider("Rainfall (mm)", 20.0, 300.0, 202.9)

    st.markdown("---")
    
    if st.button("Generate Recommendation", use_container_width=True):
        with st.spinner("🧠 AI Model is processing the parameters..."):
            time.sleep(1)
            input_features = np.array([[N, P, K, temp_manual, humidity_manual, ph, rainfall_manual]])
            scaled_features = scaler.transform(input_features)
            prediction = model.predict(scaled_features)
            
        st.success("✅ Analysis Complete!")
        st.markdown(f"""
        <div style="background-color: #E8F5E9; padding: 30px; border-radius: 15px; border: 2px solid #2E7D32; text-align: center; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
            <h2 style="color: #1B5E20; margin-bottom: 0px;">Highly Recommended Crop:</h2>
            <h1 style="color: #2E7D32; font-size: 55px; margin-top: 10px; text-transform: uppercase; font-weight: 900;">🌾 {prediction[0]}</h1>
        </div>
        """, unsafe_allow_html=True)

# --- PORTFOLIO FOOTER ---
st.markdown("---")
st.markdown("""
    <p style='text-align: center; color: #000000 !important; font-size: 14px;'>
        <b>MCA Data Science Portfolio Project | Developed by Harpreet Singh</b>
    </p>
""", unsafe_allow_html=True)