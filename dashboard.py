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
  # Keep original Booking_ID for the final report
  booking_ids = input_df["Booking_ID"]

  # 1. Perform the same feature engineering as in the notebook
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

  # 2. Use the loaded preprocessor to transform the data
  data_processed_np = preprocessor.transform(input_df)

  # 3. Create a DataFrame with the FULL set of columns from the preprocessor
  full_processed_df = pd.DataFrame(data_processed_np, columns=preprocessor_cols)

  # 4. Select only the columns the MODEL was trained on (28 columns)
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

uploaded_file = st.file_uploader(
  "Choose a CSV file of new bookings", type="csv"
)

if uploaded_file is not None:
  # Initialize session state if it doesn't exist to store results across reruns
  if "prediction_results" not in st.session_state:
    st.session_state.prediction_results = None
  if "last_file_identifier" not in st.session_state:
    st.session_state.last_file_identifier = None

  current_file_identifier = (uploaded_file.name, uploaded_file.size)

  # If a new file is uploaded, clear old results to avoid confusion
  if st.session_state.last_file_identifier != current_file_identifier:
    st.session_state.prediction_results = None
    st.session_state.last_file_identifier = current_file_identifier

  # Load the data
  daily_data = pd.read_csv(uploaded_file)
  st.write("---")
  st.subheader("Uploaded Bookings Data")
  st.dataframe(daily_data.head())

  # Make predictions when the button is clicked
  if st.button("🔍 Analyze Bookings and Find High-Risk Guests"):
    with st.spinner("Running analysis... please wait."):
      # Store results in session state to persist them
      st.session_state.prediction_results = make_prediction(daily_data)

  # Display the results section if predictions have been made
  if st.session_state.prediction_results is not None:
    prediction_results = st.session_state.prediction_results

    st.write("---")
    st.subheader("🚨 High-Risk Bookings Report")

    risk_threshold = st.slider(
      "Show bookings with risk score above:", 50, 100, 80
    )

    high_risk_df = prediction_results[
      (prediction_results["Prediction"] == "Canceled")
      & (
        prediction_results["Cancellation Risk Score"]
        .str.rstrip("%")
        .astype(float)
        >= risk_threshold
      )
    ].sort_values(by="Cancellation Risk Score", ascending=False)

    # Display key metrics
    col1, col2 = st.columns(2)
    col1.metric("Total Bookings Analyzed", len(daily_data))
    col2.metric(
      "High-Risk Bookings Found",
      len(high_risk_df),
      help="These are bookings predicted to be canceled.",
    )

    # --- HISTOGRAM OF RISK SCORES ---
    st.write("---")
    st.subheader("Distribution of Risk Scores")

    # Convert risk score to numeric for plotting.
    hist_data = prediction_results.copy()
    hist_data["risk_score_numeric"] = (
        hist_data["Cancellation Risk Score"].str.rstrip("%").astype(float)
    )

    chart = (
        alt.Chart(hist_data)
        .mark_bar()
        .encode(
            alt.X(
                "risk_score_numeric:Q",
                bin=alt.Bin(step=5),
                title="Cancellation Risk Score (%)",
            ),
            alt.Y("count()", title="Number of Bookings"),
            tooltip=["count()"],
        )
        .properties(title="Distribution of Cancellation Risk Scores Across All Bookings")
    )
    st.altair_chart(chart, use_container_width=True)


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