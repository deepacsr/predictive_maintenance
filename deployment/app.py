import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

import streamlit as st

# Below commented code created for debugging
#st.title("Test App Running")
#st.write("If you see this, Docker is fine.")

# Download and load the trained model that was aved
model_path = hf_hub_download(repo_id="deepacsr/predictive-maintenance", filename="best_package_prediction_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI
st.title("Predictive Maintenance")
st.write("""
This application predicts if Engine is likely to go Faulty or is in Normal Working condition based on Sensor data.
""")

# For numerical variables, Min and Max value taken based on the current data available.
# Default value set at mean value
EngineRPM = st.number_input("Engine RPM", min_value=60, max_value=2240, value=791, step =50)
LubricanOilPressure = st.number_input("Lubricant oil pressure", min_value=0.003, max_value=7.2, value=3.3, step =0.1)
FuelPressure = st.number_input("Fuel Pressure", min_value=0.003, max_value=21.1, value=6.6, step =0.1)
CoolantPressure = st.number_input("Coolant Pressure", min_value=0.0024, max_value=7472.0, value=2.33, step =0.1)
LubricantOilTemp = st.number_input("Lubricant Oil Temp", min_value=71.3, max_value=89.58, value=77.64, step =1.0)
CoolantTemp = st.number_input("Coolant Temp", min_value=61.67, max_value=195.52, value=78.42, step =1.0)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    "Engine rpm": EngineRPM,
    "Lub oil pressure": LubricanOilPressure,
    "Fuel pressure": FuelPressure,
    "Coolant pressure": CoolantPressure,
    "lub oil temp": LubricantOilTemp,
    "Coolant temp": CoolantTemp
}])


# Predict button

if st.button("Predict Maintenance"):
    prediction = model.predict(input_data)[0]
    # Model predicts based on the input value

    st.subheader("Prediction Result:")

    if prediction == 0:
        st.success("Engine is in normal conmdition")
    else:
        st.warning("Engine Likely to Fail")
