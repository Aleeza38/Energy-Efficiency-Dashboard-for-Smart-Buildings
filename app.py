# app.py
"""
🏙️ Smart Building Energy Efficiency Dashboard
- Predict Heating & Cooling Loads
- Color-coded predictions
- Interactive visualizations
- Single & Bulk predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
import os
import plotly.express as px
import tensorflow as tf # Keep general tensorflow import for other potential uses
import tensorflow.keras.layers as layers_keras # Direct import for BatchNormalization

st.set_page_config(page_title="🏙️ Smart Building Energy Dashboard", layout="wide")
st.title("🏙️ Smart Building Energy Efficiency Dashboard")
st.markdown("Predict **Heating** and **Cooling** Loads for buildings using AI models.")

# ---------------------------
# 1️⃣ Load Models and Assets
# ---------------------------
try:
    scaler = joblib.load('feature_scaler.joblib')
    lr_heat = joblib.load('lr_heat_model.joblib')
    rf_heat = joblib.load('rf_heat_model.joblib')
    lr_cool = joblib.load('lr_cool_model.joblib')
    rf_cool = joblib.load('rf_cool_model.joblib')
    # Custom objects dictionary for loading models with BatchNormalization
    custom_objects = {'BatchNormalization': layers_keras.BatchNormalization}
    model_heat_dl = load_model('heating_model.keras', custom_objects=custom_objects)
    model_cool_dl = load_model('cooling_model.keras', custom_objects=custom_objects)
    summary = pd.read_csv('model_summary.csv')
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

FEATURE_COLS = [
    'Relative_Compactness','Surface_Area','Wall_Area','Roof_Area',
    'Overall_Height','Orientation','Glazing_Area','Glazing_Area_Distribution'
]

# ---------------------------
# 2️⃣ Sidebar Navigation
# ---------------------------
section = st.sidebar.radio("Navigate", ["Single Prediction", "Bulk Prediction", "Model Performance", "Feature Importance"])

# ---------------------------
# Helper Functions
# ---------------------------
def color_code_load(load, thresholds=[20,30]):
    if load < thresholds[0]:
        return f"<span style='color:green;font-weight:bold'>{load:.2f}</span>"
    elif load < thresholds[1]:
        return f"<span style='color:orange;font-weight:bold'>{load:.2f}</span>"
    else:
        return f"<span style='color:red;font-weight:bold'>{load:.2f}</span>"

def show_alerts(heat, cool):
    if heat > 35: st.warning("⚠️ Heating load is very high! Consider optimizing building design.")
    elif heat < 15: st.success("✅ Heating load is low and efficient.")
    if cool > 35: st.warning("⚠️ Cooling load is very high! Consider insulation or shading.")
    elif cool < 15: st.success("✅ Cooling load is low and efficient.")

def get_model(choice):
    if choice=='Deep Learning': return model_heat_dl, model_cool_dl, True
    elif choice=='RandomForest': return rf_heat, rf_cool, False
    else: return lr_heat, lr_cool, False

# ---------------------------
# 3️⃣ Single Prediction
# ---------------------------
if section=="Single Prediction":
    st.header("🔢 Single Building Prediction")
    col1, col2 = st.columns(2)
    with col1:
        relative_compactness = st.number_input("Relative Compactness",0.0,1.0,0.7,0.01, help="Ratio of building volume to surface area. Higher is better.")
        surface_area = st.number_input("Surface Area",0.0,2000.0,650.0, step=1.0)
        wall_area = st.number_input("Wall Area",0.0,1000.0,300.0, step=1.0)
        roof_area = st.number_input("Roof Area",0.0,500.0,150.0, step=1.0)
    with col2:
        overall_height = st.number_input("Overall Height",0.0,20.0,5.0, step=0.1)
        orientation = st.selectbox("Orientation",[2,3,4,5])
        glazing_area = st.selectbox("Glazing Area",[0.0,0.1,0.25,0.4])
        glazing_area_distribution = st.selectbox("Glazing Distribution",[0,1,2,3,4,5])

    model_choice = st.selectbox("Select Model",['Deep Learning','RandomForest','LinearRegression'])

    if st.button("Predict"):
        df_input = pd.DataFrame([[relative_compactness,surface_area,wall_area,roof_area,
                                  overall_height,orientation,glazing_area,glazing_area_distribution]],
                                columns=FEATURE_COLS)
        scaled = scaler.transform(df_input)
        heat_model, cool_model, is_dl = get_model(model_choice)
        heat = heat_model.predict(scaled).flatten()[0] if is_dl else heat_model.predict(scaled)[0]
        cool = cool_model.predict(scaled).flatten()[0] if is_dl else cool_model.predict(scaled)[0]

        st.markdown(f"🔥 Heating Load: {color_code_load(heat)} kWh/m²", unsafe_allow_html=True)
        st.markdown(f"❄️ Cooling Load: {color_code_load(cool)} kWh/m²", unsafe_allow_html=True)
        show_alerts(heat, cool)

# ---------------------------
# 4️⃣ Bulk Prediction
# ---------------------------
elif section=="Bulk Prediction":
    st.header("📊 Bulk Prediction")
    uploaded_file = st.file_uploader("Upload CSV/Excel", type=["csv","xlsx"])
    model_choice = st.selectbox("Select Model for Bulk Prediction", ['Deep Learning','RandomForest','LinearRegression'])

    if uploaded_file:
        ext = uploaded_file.name.split(".")[-1]
        df = pd.read_csv(uploaded_file) if ext=="csv" else pd.read_excel(uploaded_file)
        missing = [col for col in FEATURE_COLS if col not in df.columns]
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
        else:
            X = scaler.transform(df[FEATURE_COLS])
            heat_model, cool_model, is_dl = get_model(model_choice)
            df['Pred_Heating'] = heat_model.predict(X).flatten() if is_dl else heat_model.predict(X)
            df['Pred_Cooling'] = cool_model.predict(X).flatten() if is_dl else cool_model.predict(X)

            st.subheader("Preview of Predictions")
            st.dataframe(df.head())

            # Color-coded heatmap
            st.subheader("🔥 Cooling & Heating Load Heatmap")
            fig = px.imshow(df[['Pred_Heating','Pred_Cooling']], text_auto=True,
                            labels=dict(x="Load Type", y="Building Index", color="kWh/m²"),
                            aspect="auto", color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)

            st.download_button("⬇️ Download Predictions", df.to_csv(index=False).encode('utf-8'), "predictions.csv")

            # Actual vs predicted if actual values exist
            if 'Heating_Load' in df.columns and 'Cooling_Load' in df.columns:
                st.subheader("Actual vs Predicted Comparison")
                col1, col2 = st.columns(2)
                with col1:
                    fig, ax = plt.subplots()
                    sns.scatterplot(x=df['Heating_Load'], y=df['Pred_Heating'], alpha=0.6, ax=ax)
                    ax.plot([df['Heating_Load'].min(), df['Heating_Load'].max()],
                            [df['Heating_Load'].min(), df['Heating_Load'].max()], 'k--', lw=2)
                    ax.set_xlabel('Actual Heating')
                    ax.set_ylabel('Predicted Heating')
                    st.pyplot(fig)
                with col2:
                    fig, ax = plt.subplots()
                    sns.scatterplot(x=df['Cooling_Load'], y=df['Pred_Cooling'], alpha=0.6, ax=ax)
                    ax.plot([df['Cooling_Load'].min(), df['Cooling_Load'].max()],
                            [df['Cooling_Load'].min(), df['Cooling_Load'].max()], 'k--', lw=2)
                    ax.set_xlabel('Actual Cooling')
                    ax.set_ylabel('Predicted Cooling')
                    st.pyplot(fig)

# ---------------------------
# 5️⃣ Model Performance
# ---------------------------
elif section=="Model Performance":
    st.header("📈 Model Performance Metrics")
    st.dataframe(summary)
    st.subheader("R² Comparison")
    st.bar_chart(summary, x='Model', y='R2', use_container_width=True)

# ---------------------------
# 6️⃣ Feature Importance
# ---------------------------
elif section=="Feature Importance":
    st.header("🌟 Random Forest Feature Importance")
    for target in ['Heating','Cooling']:
        path = f'images/{target}_feature_importance.png'
        if os.path.exists(path):
            st.subheader(f"{target} Load")
            st.image(path)
