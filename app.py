# app.py
import os
import base64
import pickle as pk
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.exceptions import NotFittedError

# -------------------- App Config --------------------
st.set_page_config(
    page_title="Fuel Consumption Analysis",
    page_icon="⛽",
    layout="centered",
    initial_sidebar_state="expanded",
)

# -------------------- CSS / UI --------------------
PAGE_BG = """
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

html, body, [class*="css"]  {{
    font-family: 'Poppins', sans-serif;
}}

/* background on main container */
[data-testid="stAppViewContainer"] {{
    background: url("https://wallpapercave.com/wp/wp6696562.jpg");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}}

/* make header transparent */
[data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
}}

/* Glass card container for block content */
.block-container {{
    background: rgba(0, 0, 0, 0.55);
    padding: 2rem 2.5rem;
    border-radius: 18px;
    backdrop-filter: blur(10px);
    color: white;
}}

/* Headings */
h1, h2, h3, h4, h5, h6 {{
    color: white !important;
    text-align: center;
}}

/* Labels and inputs */
label, .stMarkdown, .css-1t3w2i1 {{
    color: #ffffff !important;
    font-weight: 600 !important;
}

/* Button styling */
.stButton>button {{
    background-color: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.18);
    color: white;
    padding: 0.6rem 1.2rem;
    font-weight: 600;
    border-radius: 10px;
    backdrop-filter: blur(5px);
    transition: 0.2s ease;
}}
.stButton>button:hover {{
    background-color: rgba(255,255,255,0.18);
}}

/* Inputs look */
.stNumberInput input, .stSelectbox select {{
    color: white !important;
    background: rgba(255,255,255,0.04) !important;
    border-radius: 6px;
}}

.small-muted {{
    color: rgba(255,255,255,0.7);
    font-size: 0.9rem;
}}
</style>
"""

st.markdown(PAGE_BG, unsafe_allow_html=True)

# -------------------- Helpers & Model Loading --------------------
MODEL_PATH = Path("trained_model_lr.sav")
SCALE_PATH = Path("scaled_data.sav")


@st.cache_data(show_spinner=False)
def load_pickle(path: Path):
    """Safely load pickle file; returns None on failure."""
    try:
        with open(path, "rb") as f:
            obj = pk.load(f)
        return obj
    except FileNotFoundError:
        return None
    except Exception as e:
        # keep the error trace short in cache; actual message shown in UI
        return {"__load_error__": str(e)}


loaded_model = load_pickle(MODEL_PATH)
scaled_data = load_pickle(SCALE_PATH)


def model_ready():
    if loaded_model is None or isinstance(loaded_model, dict) and "__load_error__" in loaded_model:
        return False
    if scaled_data is None or isinstance(scaled_data, dict) and "__load_error__" in scaled_data:
        return False
    return True


def load_error_messages():
    msgs = []
    if loaded_model is None:
        msgs.append(f"- Model file not found at `{MODEL_PATH}`.")
    elif isinstance(loaded_model, dict) and "__load_error__" in loaded_model:
        msgs.append(f"- Model load error: {loaded_model['__load_error__']}")
    if scaled_data is None:
        msgs.append(f"- Scaler file not found at `{SCALE_PATH}`.")
    elif isinstance(scaled_data, dict) and "__load_error__" in scaled_data:
        msgs.append(f"- Scaler load error: {scaled_data['__load_error__']}")
    return "\n".join(msgs) if msgs else None


# -------------------- Input conversion & prediction --------------------
VEHICLE_CLASSES = [
    'Two-seater','Minicompact','Compact','Subcompact','Mid-size','Full-size',
    'SUV: Small','SUV: Standard','Minivan','Station wagon: Small','Station wagon: Mid-size',
    'Pickup truck: Small','Special purpose vehicle','Pickup truck: Standard'
]
TRANSMISSION = ['AV','AM','M','AS','A']
FUEL = ["D", "E", "X", "Z"]  # encoding will be one-hot for fuel

def input_converter(inp):
    """
    Accepts a list with elements:
    [Vehicle_class(str), Engine_size(int), Cylinders(int), Transmission(str), CO2_Rating(int), Fuel_type(str)]
    Returns the scaled array and prediction string.
    """
    # Defensive check
    if not model_ready():
        raise FileNotFoundError("Model or scaler not loaded. See app messages for details.")

    vcl = VEHICLE_CLASSES
    trans = TRANSMISSION
    fuel = FUEL
    lst = []
    # expect input in specific order
    for i in range(6):
        val = inp[i]
        if isinstance(val, str):
            if val in vcl:
                lst.append(vcl.index(val))
            elif val in trans:
                lst.append(trans.index(val))
            elif val in fuel:
                # expand fuel into one-hot (D, E, X, Z)
                one_hot = [0, 0, 0, 0]
                idx = fuel.index(val)
                one_hot[idx] = 1
                lst.extend(one_hot)
                # fuel uses up slot(s) so continue loop properly; note we break from loop's fuel handling by not appending original
        else:
            # numeric
            lst.append(val)

    # Check length: expected length depends on how fuel was encoded.
    arr = np.asarray(lst).reshape(1, -1)
    try:
        arr_scaled = scaled_data.transform(arr)
    except Exception as e:
        raise NotFittedError(f"Scaler error: {e}")

    try:
        pred = loaded_model.predict(arr_scaled)
    except Exception as e:
        raise RuntimeError(f"Model prediction error: {e}")

    return pred[0]


# -------------------- UI Pages --------------------
def page_home():
    st.markdown("<h1>Fuel Consumption Analysis</h1>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="small-muted">
        This app predicts vehicle fuel consumption (L/100km) from a small set of vehicle features.
        Use the **Predict** page to try the model or explore the repository for full analysis and notebooks.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### Project Snapshot")
        st.write(
            """
            - **Model:** Linear Regression (pretrained)
            - **Inputs:** Vehicle class, Engine size, Cylinders, Transmission, CO2 rating, Fuel type
            - **Output:** Fuel consumption in L/100km
            """
        )
        
    with col2:
        st.markdown("### Model Status")
        if model_ready():
            st.success("Model and scaler loaded successfully ✅")
            st.write("Model file:", MODEL_PATH.name)
            st.write("Scaler file:", SCALE_PATH.name)
        else:
            st.error("Model or scaler not loaded ❌")
            msg = load_error_messages()
            if msg:
                st.markdown("**Load errors:**")
                st.code(msg)


def page_predict():
    st.markdown("<h2>Predict Fuel Consumption</h2>", unsafe_allow_html=True)
    if not model_ready():
        st.error("Model/scaler not available. Please check the Home page for load errors.")
        return

    with st.form(key="predict_form"):
        # Use columns for nicer layout
        c1, c2 = st.columns(2)
        with c1:
            Vehicle_class = st.selectbox("Vehicle class", options=VEHICLE_CLASSES)
            Engine_size = st.slider("Engine Size (1 - 7)", min_value=1, max_value=7, value=3)
            Cylinders = st.number_input("Cylinders (1 - 16)", min_value=1, max_value=16, value=4, step=1)
        with c2:
            Transmission = st.selectbox("Transmission", options=TRANSMISSION)
            Co2_Rating = st.slider("CO2 Rating (1 - 10)", min_value=1, max_value=10, value=5)
            Fuel_type = st.selectbox("Fuel type", options=FUEL)

        submitted = st.form_submit_button("Predict 🔍")

    if submitted:
        try:
            inp = [Vehicle_class, Engine_size, Cylinders, Transmission, Co2_Rating, Fuel_type]
            pred_value = input_converter(inp)
            st.markdown(f"<h3 style='text-align:center;color:#a8ffb0;'>Predicted Fuel Consumption: {round(pred_value, 2)} L/100km</h3>", unsafe_allow_html=True)

            # Show a little breakdown table of inputs and model readiness
            df = pd.DataFrame({
                "Feature": ["Vehicle class", "Engine size", "Cylinders", "Transmission", "CO2 Rating", "Fuel type"],
                "Value": [Vehicle_class, Engine_size, Cylinders, Transmission, Co2_Rating, Fuel_type]
            })
            st.table(df)
        except Exception as e:
            st.error(f"Prediction failed: {e}")


def page_about():
    st.markdown("<h2>About this App</h2>", unsafe_allow_html=True)
    st.markdown(
        """
        """
    )
    st.markdown("### Planned improvements")
    st.write(
        """
        - Add more models (Random Forest, XGBoost) and compare.  
        - Host a small web demo (Flask/Streamlit + Heroku/Render).  
        - Add CI to auto-run tests and linting.
        """
    )


# -------------------- Sidebar Navigation --------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Predict", "About"])

# Optionally show a small help / repo link area
st.sidebar.markdown("---")
st.sidebar.markdown("**Files expected**")
st.sidebar.markdown(f"- `{MODEL_PATH.name}` (pickle)\n- `{SCALE_PATH.name}` (picker/scaler)")
st.sidebar.markdown("---")
st.sidebar.markdown("Need help? Add files or check paths in the app folder.")

# -------------------- Run selected page --------------------
if page == "Home":
    page_home()
elif page == "Predict":
    page_predict()
elif page == "About":
    page_about()
else:
    st.info("Choose a page from the sidebar.")
