import numpy as np
import pandas as pd
import pickle as pk
import streamlit as st
import base64
import sklearn as sk
from streamlit.components.v1 import html

# Correct file paths (files are at repo root)
loaded_model = pk.load(open("trained_model_lr.sav", "rb"))
scaled_data = pk.load(open("scaled_data.sav", "rb"))

@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://wallpapercave.com/wp/wp6696562.jpg");
background-size: 100%;
background-position: center;
background-repeat: no-repeat;
background-attachment: fixed;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

def input_converter(inp):
    vcl = ['Two-seater', 'Minicompact', 'Compact', 'Subcompact', 'Mid-size', 'Full-size',
           'SUV: Small', 'SUV: Standard', 'Minivan', 'Station wagon: Small',
           'Station wagon: Mid-size', 'Pickup truck: Small', 'Special purpose vehicle',
           'Pickup truck: Standard']
    trans = ['AV','AM','M','AS','A']
    fuel = ["D", "E", "X", "Z"]
    lst = []
    for i in range(6):
        if isinstance(inp[i], str):
            if inp[i] in vcl:
                lst.append(vcl.index(inp[i]))
            elif inp[i] in trans:
                lst.append(trans.index(inp[i]))
            elif inp[i] in fuel:
                one_hot = [0, 0, 0, 0]
                one_hot[fuel.index(inp[i])] = 1
                lst.extend(one_hot)
                break
        else:
            lst.append(inp[i])
    arr = np.asarray(lst).reshape(1, -1)
    arr = scaled_data.transform(arr)
    prediction = loaded_model.predict(arr)
    return f"The Fuel Consumption L/100km is {round(prediction[0], 2)}"

def main():
    st.markdown("<h1 style='text-align: center; color: white;'>Fuel Consumption Prediction</h1>", unsafe_allow_html=True)

    vehicle = ['Two-seater','Minicompact','Compact','Subcompact','Mid-size','Full-size',
               'SUV: Small','SUV: Standard','Minivan','Station wagon: Small',
               'Station wagon: Mid-size','Pickup truck: Small','Special purpose vehicle',
               'Pickup truck: Standard']
    transmission = ['AV','AM','M','AS','A']
    fuel = ["D", "E", "X", "Z"]

    Vehicle_class = st.selectbox("Enter Vehicle class", vehicle)
    Engine_size = st.selectbox("Select Engine Size [1-7]", [1,2,3,4,5,6,7])
    Cylinders = st.number_input("Enter number of Cylinders [1-16]", min_value=1, max_value=16)
    Transmission = st.selectbox("Select the Transmission", transmission)
    Co2_Rating = st.number_input("Enter CO2 Rating [1-10]", min_value=1, max_value=10)
    Fuel_type = st.selectbox("Select the Fuel type", fuel)

    if st.button("Predict 🔍"):
        result = input_converter([Vehicle_class, Engine_size, Cylinders, Transmission, Co2_Rating, Fuel_type])
        st.markdown(f"<h2 style='color:white;'><b>{result}</b>!</h2>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
