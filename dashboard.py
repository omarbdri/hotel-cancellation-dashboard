import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import altair as alt

# --- IMPORTANT ---
# This file is the main dashboard for the hotel booking cancellation prediction app.
# It uses the pre-trained model and supporting assets to make predictions on new bookings.
# The app allows staff to upload a CSV file of new bookings and get a report of high-risk bookings.
# To run the app, you need to have the following files in the same directory:
# - hotel_cancellation_model.joblib
# - hotel_data_preprocessor.joblib
# - preprocessor_output_columns.joblib
# - model_columns.joblib
# - dashboard.py
# then run the command: streamlit run dashboard.py

# --- CONFIGURATION ---
st.set_page_config(
  page_title="High-Risk Booking Detector",
  page_icon="🏨",
  layout="wide",
)

# --- LOAD PRODUCTION ASSETS ---
@st.cache_resource
def load_assets():
  """Loads the pre-trained model and supporting assets."""
  model = joblib.load("hotel_cancellation_model.joblib")
  preprocessor = joblib.load("hotel_data_preprocessor.joblib")
  preprocessor_cols = joblib.load("preprocessor_output_columns.joblib")
  model_cols = joblib.load("model_columns.joblib")
  return model, preprocessor, preprocessor_cols, model_cols


model, preprocessor, preprocessor_cols, model_cols = load_assets()


# --- HELPER FUNCTION (UPDATED) ---
def make_prediction(input_df):
  """
    Takes a dataframe of new bookings, preprocesses it, and returns
    predictions with risk scores.
    """
  # Use the dataframe's index as booking ID since we assume Booking_ID column won't be in the data
  booking_ids = input_df.index

  # 1. Perform feature engineering for 'is_weekend_arrival' if it's not already present
  if 'is_weekend_arrival' not in input_df.columns:
    date_cols = ["arrival_year", "arrival_month", "arrival_date"]
    if all(col in input_df.columns for col in date_cols):
        input_df["is_weekend_arrival"] = (
            pd.to_datetime(
                dict(
                    year=input_df["arrival_year"],
                    month=input_df["arrival_month"],
                    day=input_df["arrival_date"],
                ),
                errors="coerce",
            )
            .dt.dayofweek.isin([5, 6])
            .astype(int)
        )
    else:
        st.warning(
            "The 'is_weekend_arrival' column was not found and could not be calculated "
            "from date columns. Setting it to a default value (0). "
            "This may impact prediction accuracy."
        )
        input_df["is_weekend_arrival"] = 0

  # 2. Use the loaded preprocessor to transform the data, or use data as is if pre-processed
  # Check if the data seems to be already one-hot encoded
  if set(model_cols).issubset(input_df.columns):
      st.info("Input data appears to be pre-processed. Skipping the pre-processing step.")
      data_for_model = input_df[model_cols]
  else:
      st.info("Running data through the pre-processor.")
      data_processed_np = preprocessor.transform(input_df)
      full_processed_df = pd.DataFrame(data_processed_np, columns=preprocessor_cols)
      data_for_model = full_processed_df[model_cols]

  # 5. Make predictions on the correctly shaped data
  predictions = model.predict(data_for_model)
  pred_probs = model.predict_proba(data_for_model)[:, 0]

  # 6. Create the final report dataframe
  results_df = pd.DataFrame(
    {
      "Booking_ID": booking_ids,
      "Prediction": ["Canceled" if p == 0 else "Not Canceled" for p in predictions],
      "Cancellation Risk Score": [f"{prob:.0%}" for prob in pred_probs],
      "Lead Time": input_df["lead_time"],
      "Avg Price Per Room": input_df["avg_price_per_room"],
      "Special Requests": input_df["no_of_special_requests"],
    }
  )

  return results_df


# --- STREAMLIT UI ---
st.title("🏨 Daily High-Risk Booking Report")
st.markdown(
  """
    Upload the CSV file with today's new bookings to identify which ones are at a high risk of cancellation.
    The staff can use this report to proactively contact guests and offer assistance.
    """
)

# --- Data Loading and Analysis ---
# Initialize session state for caching results
if "prediction_results" not in st.session_state:
    st.session_state.prediction_results = None
if "last_file_identifier" not in st.session_state:
    st.session_state.last_file_identifier = None
if "daily_data" not in st.session_state:
    st.session_state.daily_data = None

# File uploader allows users to upload their own data
uploaded_file = st.file_uploader(
    "Upload a new CSV file to override the default report", type="csv"
)

# Determine the data source: uploaded file or the default file
data_source = uploaded_file
source_id = None
if data_source:
    source_id = (data_source.name, data_source.size)
    # If a new file is uploaded, clear old results.
    if st.session_state.last_file_identifier != source_id:
        st.session_state.prediction_results = None
else:
    # If no file is uploaded, try to use the default one.
    try:
        data_source = "cleaned_data.csv"
        source_id = "default"
    except FileNotFoundError:
        st.info("`cleaned_data.csv` not found. Please upload a file to begin.")
        data_source = None

# If we have a data source and no results are cached for it, run the analysis.
if data_source and st.session_state.last_file_identifier != source_id:
    with st.spinner("Loading data and running analysis..."):
        daily_data = pd.read_csv(data_source)
        st.session_state.daily_data = daily_data
        st.session_state.prediction_results = make_prediction(daily_data)
        st.session_state.last_file_identifier = source_id
        if uploaded_file is None:
            st.toast("Loaded default report from `cleaned_data.csv`.", icon="✅")


# Display the results if they exist in the session state
if st.session_state.prediction_results is not None:
    prediction_results = st.session_state.prediction_results
    daily_data = st.session_state.daily_data
    st.write("---")
    st.subheader("🚨 High-Risk Bookings Report")

    high_risk_df = prediction_results[
        (prediction_results["Prediction"] == "Canceled")
    ].sort_values(by="Cancellation Risk Score", ascending=False)

    # Display key metrics
    col1, col2 = st.columns(2)
    col1.metric("Total Bookings Analyzed", len(daily_data))
    col2.metric(
      "High-Risk Bookings Found",
      len(high_risk_df),
      help="These are bookings predicted to be canceled.",
    )

    # Display the actionable table
    st.subheader("Actionable High-Risk Bookings List")
    st.dataframe(high_risk_df)

    # Allow downloading the report
    csv = high_risk_df.to_csv(index=False).encode("utf-8")
    st.download_button(
      label="📥 Download High-Risk Report as CSV",
      data=csv,
      file_name=f"high_risk_report_{datetime.now().strftime('%Y%m%d')}.csv",
      mime="text/csv",
    )