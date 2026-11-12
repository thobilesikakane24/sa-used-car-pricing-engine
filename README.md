# 🚗 Predictive Analytics & Structural Price Risks in the South African Used Car Market

**A Machine Learning Valuation Engine for Pre-Owned Dealerships**  
*Zaio Institute of Technology — Data Science Capstone Project*

[![Streamlit App](https://streamlit.io)](https://sa-used-car-pricing-engine-naavuxbtsapmujhn4z7bsi.streamlit.app/)
[![Python](https://shields.io)](https://python.org)
[![Scikit-Learn](https://shields.io)](https://scikit-learn.org)

🔗 **Live Interactive App:** [Launch the SA Used Car Pricing Engine](https://sa-used-car-pricing-engine-naavuxbtsapmujhn4z7bsi.streamlit.app/)

---

## 📌 Project Overview

The South African pre-owned automotive market is experiencing major structural shifts. An influx of affordable, warranty-backed Chinese imports (such as Chery, Haval, and GWM) has compressed the price gap between new and used vehicles. At the same time, traditional linear depreciation rules fail to account for critical local risk factors like manufacturer exits, parts scarcity, and high-theft/hijacking exposure.

This project delivers an end-to-end machine learning regression engine and an interactive **Streamlit** web application designed to help South African dealership procurement and sales teams accurately forecast vehicle resale values in **South African Rand (ZAR)**.

---

## 🎯 Research Objectives & Key Findings

The modeling pipeline was built to answer three core market questions:

1. **Discontinuation Collapse (`is_discontinued`):** *How severely do brand exits and model retirements accelerate depreciation?*
   * **Finding:** Discontinued passenger cars (e.g., Chevrolet, Fiat, Datsun, Ford EcoSport/Figo) suffer an immediate **15%–25% valuation discount** after Year 4 due to parts scarcity, whereas high-utility diesel models retain value better through independent mechanic networks.
2. **The Hijacking Paradox (`high_demand_target`):** *Does high theft/hijacking popularity act as a price depressor or a liquidity stabilizer?*
   * **Finding:** High-target, parts-interchangeable vehicles (e.g., VW Polo, Toyota Fortuner, Toyota Innova, Suzuki Swift) retain **10%–18% higher resale value** in mid-to-late life (Years 5–10). Extreme market demand and parts liquidity outweigh insurance premium pushback.
3. **Brand Equity Shield (`Brand_Tier`):** *How does segmenting market tiers improve baseline accuracy?*
   * **Finding:** Categorizing vehicles into `Trusted_Legacy`, `Luxury`, and `Value_Challenger` reveals that `Trusted_Legacy` brands maintain flatter depreciation curves compared to `Luxury` imports, which experience steep initial value drops.

---

## 🛠️ Data Pipeline & Feature Engineering

### 1. Data Cleaning & Preprocessing
* **Currency Standardization:** Converted comprehensive used vehicle listings into South African Rand (ZAR).
* **Regex Text Extraction:** Parsed unstructured `AdditionInfo` text descriptions to recover missing attributes (`Brand`, `Model`, `Year`, `FuelType`, `Engine_Capacity`, and `Seating_Capacity`).
* **Anomaly Filtering:** Corrected sentinel year values (`9999`), stripped scrap/non-vehicle listings, and filtered extreme odometer and price outliers.
* **Unified Transformation:** Built a `ColumnTransformer` applying `StandardScaler` to numerical features and `OneHotEncoder(handle_unknown='ignore')` to categorical features.

### 2. Domain-Driven Feature Engineering
* **`Brand_Tier`:** Segments manufacturers into `Luxury`, `Trusted_Legacy`, and `Value_Challenger` to capture brand equity.
* **`is_discontinued`:** Binary flag identifying exited brands (Chevrolet, Fiat, Datsun) and retired lines (EcoSport, Figo, Civic, Alto 800).
* **`high_demand_target`:** Binary flag for high-volume, parts-interchangeable vehicles exposed to theft/hijacking demand (VW Polo, Toyota Fortuner, Innova, Swift).
* **`km_per_year`:** Engineered usage intensity metric (`kmDriven / (Age + 1)`) to distinguish high-mileage fleet vehicles from lightly driven private cars.

---

## 📈 Model Performance & Diagnostics

Models were evaluated using an **80/20 train-test split** with **3-fold cross-validation** to prevent data leakage.

| Model | R² Score | Key Observations |
| :--- | :---: | :--- |
| **Baseline Linear Regression** | `~0.65` | Struggled to capture non-linear luxury depreciation curves and structural risk interactions. |
| **Tuned Random Forest Regressor** | **`~0.88`** | Successfully modeled complex feature interactions across vehicle age, brand tier, and risk flags. |

* **Hyperparameter Optimization:** Tuned `n_estimators`, `max_depth`, `min_samples_split`, and `max_features` via `RandomizedSearchCV` to reduce RMSE and control overfitting.
* **Error Analysis:** Predictions cluster tightly along the diagonal for core dealership inventory (**R 50,000 – R 600,000**). Higher residual variance appears in ultra-luxury exotic vehicles (**> R 1.5M**) due to low sample density and unrecorded trim options.

---

## 🚀 Live Deployment & Local Setup

The end-to-end preprocessing and tuned Random Forest pipeline is serialized using `joblib` (`sa_used_car_pricer.pkl`) and deployed on **Streamlit Community Cloud**.

* **Try it online:** [https://sa-used-car-pricing-engine-naavuxbtsapmujhn4z7bsi.streamlit.app/](https://sa-used-car-pricing-engine-naavuxbtsapmujhn4z7bsi.streamlit.app/)

### Repository Structure
```text
├── .devcontainer/             # Dev container configuration for Codespaces/VS Code
├── app.py                     # Streamlit interactive valuation dashboard
├── jupyter.ipynb              # EDA, regex cleaning, feature engineering, and model tuning notebook
├── requirements.txt           # Python dependencies
├── sa_used_car_pricer.pkl     # Serialized Scikit-Learn preprocessing & Random Forest pipeline
└── README.md                  # Project documentation
```

### Run Locally
1. **Clone the repository:**
   ```bash
   git clone https://github.com
   cd sa-used-car-pricing-engine
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Streamlit app:**
   ```bash
   streamlit run app.py
   ```
   *Enter basic vehicle details in the UI—the app automatically computes `Brand_Tier`, `is_discontinued`, `high_demand_target`, and `km_per_year` in the background to generate an instant ZAR retail valuation.*

---

## 🔮 Limitations & Future Roadmap

* **Regional & OEM Coverage:** The current training dataset lacks provincial location tags (e.g., Gauteng inland vs. Coastal rust multipliers) and direct Chinese OEM listings (Chery, Haval, BAIC).
* **Live Market Integration:** Scrape and integrate live South African dealership listings (AutoTrader / Cars.co.za) alongside postal-code corrosion multipliers.
* **NLP Feature Extraction:** Extract **Full Service History (FSH)** and accident/damage indicators directly from dealer listing descriptions using NLP.
* **Automated Retraining:** Implement scheduled MLOps retraining pipelines to keep pricing aligned with shifting South African market trends.
