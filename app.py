import os
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="SA Used Car Valuation Engine", layout="centered")
st.title("🚗 South African Used Car Pricing Engine")
st.markdown("Predict fair retail valuation in **ZAR** factoring in age, mileage, brand tier, discontinuation risk, and parts-target popularity.")

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sa_used_car_pricer.pkl")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at `{MODEL_PATH}`. Please run your Jupyter Notebook to export `sa_used_car_pricer.pkl` first.")
    st.stop()

model = load_model()

col1, col2 = st.columns(2)
with col1:
    brand = st.selectbox("Brand", ['toyota', 'volkswagen', 'maruti suzuki', 'hyundai', 'ford', 'honda', 'bmw', 'mercedes-benz', 'audi', 'mahindra', 'tata', 'kia', 'renault', 'chevrolet', 'nissan'])
    refined_model = st.text_input("Model Name (e.g., polo, fortuner, swift, creta)", "polo").lower().strip()
    age = st.slider("Vehicle Age (Years)", 0, 20, 4)
    km_driven = st.number_input("Odometer Reading (km)", min_value=500, max_value=350000, value=65000, step=1000)

with col2:
    transmission = st.selectbox("Transmission", ['manual', 'automatic'])
    fuel_type = st.selectbox("Fuel Type", ['petrol', 'diesel', 'hybrid/cng'])
    owner = st.selectbox("Owner History", ['first', 'second', 'third', 'unknown'])
    engine_cap = st.number_input("Engine Capacity (L)", min_value=0.8, max_value=6.0, value=1.4, step=0.1)
    seating = st.selectbox("Seating Capacity", [2, 4, 5, 7, 8], index=2)

luxury_brands = ['bmw', 'mercedes-benz', 'audi', 'porsche', 'lexus', 'land rover', 'jaguar', 'volvo']
legacy_trusted = ['toyota', 'volkswagen', 'ford', 'honda', 'hyundai', 'kia', 'nissan', 'suzuki', 'maruti suzuki', 'mahindra']
discontinued_brands = ['chevrolet', 'fiat', 'datsun', 'mitsubishi', 'opel']
discontinued_models = ['ecosport', 'figo', 'fiesta', 'endeavour', 'civic', 'accord', 'cr-v', 'brio', 'jazz', 'alto 800', 'omni', 'sx4', 'ritz', 'zen']
high_demand_targets = ['polo', 'fortuner', 'innova', 'swift', 'dzire', 'creta', 'scorpio', 'bolero']

is_luxury = int(brand in luxury_brands)
brand_tier = 'Luxury' if is_luxury else ('Trusted_Legacy' if brand in legacy_trusted else 'Value_Challenger')
is_discontinued = int(brand in discontinued_brands or refined_model in discontinued_models)
high_demand_target = int(refined_model in high_demand_targets)
km_per_year = km_driven / (age + 1)

if st.button("Calculate Fair Retail Price (ZAR)", key="predict_btn"):
    input_df = pd.DataFrame([{
        'Year': 2024 - age,
        'Age': age,
        'kmDriven': km_driven,
        'km_per_year': km_per_year,
        'Engine_Capacity': engine_cap,
        'Seating_Capacity': seating,
        'is_luxury': is_luxury,
        'is_discontinued': is_discontinued,
        'high_demand_target': high_demand_target,
        'Brand': brand,
        'Refined_Model': refined_model,
        'Transmission': transmission,
        'Owner': owner,
        'FuelType': fuel_type,
        'Brand_Tier': brand_tier
    }])
    
   
    prediction = model.predict(input_df)[0]
    st.success(f"### Estimated Market Value: R {prediction:,.2f}")

    st.info(
        f"**Risk Profile Applied:** Tier = `{brand_tier}` | "
        f"Discontinued Risk = `{'Yes' if is_discontinued else 'No'}` | "
        f"High-Demand Target = `{'Yes' if high_demand_target else 'No'}`"
    )
