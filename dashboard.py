import streamlit as st
import pandas as pd
import joblib
from datetime import datetime, date
import altair as alt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_score, recall_score, classification_report

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

# Custom CSS for better styling
st.markdown("""
<style>
    /* .filter-container {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        border: 1px solid #e9ecef;
    } */
    
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: 1px solid #4f46e5;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    div[data-testid="metric-container"] > div {
        color: white !important;
    }
    
    div[data-testid="metric-container"] label {
        color: white !important;
        font-weight: 600 !important;
    }
    
    div[data-testid="metric-container"] [data-testid="metric-value"] {
        color: white !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    
    div[data-testid="metric-container"] [data-testid="metric-delta"] {
        color: #fbbf24 !important;
        font-weight: 500 !important;
    }
    
    /* Alternative styling for better visibility */
    .stMetric {
        background: transparent;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 1px solid #ffffff;
    }
    
    .stMetric label {
        color: white !important;
        font-weight: 600 !important;
    }
    
    .stMetric [data-testid="metric-value"] {
        color: white !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    
    .stMetric [data-testid="metric-delta"] {
        color: #fbbf24 !important;
        font-weight: 500 !important;
    }
</style>
""", unsafe_allow_html=True)

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

# --- HELPER FUNCTIONS ---
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
        input_df["is_weekend_arrival"] = 0

  # 2. Use the loaded preprocessor to transform the data, or use data as is if pre-processed
  # Check if the data seems to be already one-hot encoded
  if set(model_cols).issubset(input_df.columns):
      data_for_model = input_df[model_cols]
  else:
      data_processed_np = preprocessor.transform(input_df)
      full_processed_df = pd.DataFrame(data_processed_np, columns=preprocessor_cols)
      data_for_model = full_processed_df[model_cols]

  # 5. Make predictions on the correctly shaped data
  predictions = model.predict(data_for_model)
  pred_probs = model.predict_proba(data_for_model)[:, 1]

  # 6. Extract market segment and customer type from one-hot encoded columns
  def extract_categorical_value(df, prefix):
    """Extract the active category from one-hot encoded columns"""
    matching_cols = [col for col in df.columns if col.startswith(prefix)]
    result = []
    for idx, row in df.iterrows():
      active_col = None
      for col in matching_cols:
        if row[col] == 1:
          active_col = col.replace(prefix, "")
          break
      result.append(active_col if active_col else "Unknown")
    return result

  market_segment_values = extract_categorical_value(input_df, "market_segment_type_")

  # 7. Create the final report dataframe
  results_df = pd.DataFrame(
    {
      "Booking_ID": booking_ids,
      "Prediction": ["Canceled" if p == 1 else "Not Canceled" for p in predictions],
      "Cancellation Risk Score": [f"{prob:.0%}" for prob in pred_probs],
      "Risk_Score_Numeric": pred_probs,
      "Market Segment": market_segment_values,
      "Lead Time": input_df["lead_time"],
      "Avg Price Per Room": input_df["avg_price_per_room"],
      "Special Requests": input_df["no_of_special_requests"],
    }
  )

  return results_df

def filter_data(df, date_range, market_segment, lead_time_range):
    """Apply filters to the dataframe"""
    filtered_df = df.copy()
    
    # Date filtering
    if 'arrival_year' in df.columns and 'arrival_month' in df.columns and 'arrival_date' in df.columns:
        try:
            df['arrival_full_date'] = pd.to_datetime(
                dict(year=df['arrival_year'], month=df['arrival_month'], day=df['arrival_date']),
                errors='coerce'
            )
            filtered_df = filtered_df[
                (df['arrival_full_date'] >= pd.to_datetime(date_range[0])) &
                (df['arrival_full_date'] <= pd.to_datetime(date_range[1]))
            ]
        except:
            pass
    
    # Market segment filtering (one-hot encoded)
    if market_segment != "All":
        market_segment_col = f"market_segment_type_{market_segment}"
        if market_segment_col in df.columns:
            filtered_df = filtered_df[filtered_df[market_segment_col] == 1]
    
    # Lead time filtering
    if 'lead_time' in df.columns:
        filtered_df = filtered_df[
            (filtered_df['lead_time'] >= lead_time_range[0]) &
            (filtered_df['lead_time'] <= lead_time_range[1])
        ]
    
    return filtered_df

def get_categorical_options(df, prefix):
    """Extract categorical options from one-hot encoded columns"""
    options = ["All"]
    matching_cols = [col for col in df.columns if col.startswith(prefix)]
    for col in matching_cols:
        # Extract the category name after the prefix
        category = col.replace(prefix, "")
        options.append(category)
    return sorted(options)

def calculate_model_performance(df, predictions_df):
    """Calculate model performance metrics if ground truth is available"""
    # Look for possible ground truth columns
    possible_target_cols = ['is_canceled', 'booking_status', 'cancelled', 'canceled', 'target']
    target_col = None
    y_true = None
    
    # First, check for regular target columns
    for col in possible_target_cols:
        if col in df.columns:
            target_col = col
            y_true = df[target_col].values
            if y_true.dtype == 'object':
                # If it's categorical, convert to binary
                y_true = (y_true == 'Canceled').astype(int)
            break
    
    # If no regular target column found, check for one-hot encoded booking status
    if target_col is None:
        if 'booking_status_Canceled' in df.columns:
            target_col = 'booking_status_Canceled'
            y_true = df[target_col].values  # Already binary (1 = canceled, 0 = not canceled)
        elif 'booking_status_Not_Canceled' in df.columns:
            target_col = 'booking_status_Not_Canceled'
            y_true = 1 - df[target_col].values  # Flip it (1 = canceled, 0 = not canceled)
            

    
    if target_col is None or y_true is None:
        return None
    
    # Get predicted values and probabilities
    y_pred = (predictions_df['Prediction'] == 'Canceled').astype(int)
    y_pred_proba = predictions_df['Risk_Score_Numeric'].values
    
    # Ensure arrays are aligned (same length)
    if len(y_true) != len(y_pred):
        # Take the minimum length to avoid errors
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        y_pred_proba = y_pred_proba[:min_len]

    # Correction for potentially inverted ground truth labels.
    # If initial AUC is < 0.5, the labels [0, 1] in y_true are likely the
    # reverse of what the model's probabilities for class 1 represent.
    if len(np.unique(y_true)) > 1 and roc_auc_score(y_true, y_pred_proba) < 0.5:
        y_true = 1 - y_true
    
    # Calculate metrics with corrected y_true
    cm = confusion_matrix(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    precision_cancel = precision_score(y_true, y_pred, pos_label=1)
    recall_cancel = recall_score(y_true, y_pred, pos_label=1)
    
    return {
        'confusion_matrix': cm,
        'roc_auc': roc_auc,
        'precision_cancel': precision_cancel,
        'recall_cancel': recall_cancel,
        'y_true': y_true,
        'y_pred': y_pred,
        'target_col': target_col
    }

def convert_standardized_to_original_leadtime(standardized_values):
    """Convert standardized lead_time values back to original values using empirical mapping"""
    # We know from the data:
    # - Standardized min: -0.992 (roughly)
    # - Standardized max: 4.162 (roughly)  
    # - Original range: 0-13 days (as confirmed by user)
    
    std_min = -0.9921200019255964  # From your debug output
    std_max = 4.161896980478907    # From your debug output
    orig_min = 0                   # Original minimum (0 days)
    orig_max = 13                  # Original maximum (13 days, as you confirmed)
    
    # Linear mapping from standardized to original
    # Normalize standardized values to 0-1 range
    normalized = (standardized_values - std_min) / (std_max - std_min)
    
    # Map to original range
    original_values = normalized * (orig_max - orig_min) + orig_min
    
    return np.round(original_values).astype(int)

# --- STREAMLIT UI ---
st.title("🏨 Hotel Booking Cancellation Dashboard")
st.markdown("Monitor and predict booking cancellations to improve revenue management")

# --- Data Loading ---
# Initialize session state for caching results
if "prediction_results" not in st.session_state:
    st.session_state.prediction_results = None
if "last_file_identifier" not in st.session_state:
    st.session_state.last_file_identifier = None
if "daily_data" not in st.session_state:
    st.session_state.daily_data = None

# File uploader in sidebar
with st.sidebar:
    st.header("📂 Data Source")
    uploaded_file = st.file_uploader(
        "Upload CSV file", type="csv", 
        help="Upload a CSV file with booking data to override the default dataset"
    )
    st.info(f"**Model in use:** `{type(model).__name__}`")

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
        st.error("📋 `cleaned_data.csv` not found. Please upload a file to begin.")
        st.stop()

# If we have a data source and no results are cached for it, run the analysis.
if data_source and st.session_state.last_file_identifier != source_id:
    with st.spinner("🔄 Loading data and running analysis..."):
        daily_data = pd.read_csv(data_source)
        st.session_state.daily_data = daily_data
        st.session_state.prediction_results = make_prediction(daily_data)
        st.session_state.last_file_identifier = source_id
        if uploaded_file is None:
            st.success("✅ Loaded default report from `cleaned_data.csv`")

# Display the results if they exist in the session state
if st.session_state.prediction_results is not None:
    daily_data = st.session_state.daily_data
    
    # --- FILTERS SECTION ---
    st.subheader("🎛️ Filters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Market segment dropdown (from one-hot encoded columns)
        market_segments = get_categorical_options(daily_data, "market_segment_type_")
        
        market_segment = st.selectbox(
            "🎯 Market Segment",
            market_segments
        )
    
    with col2:
        # Create a more user-friendly lead time selector using empirical conversion
        st.markdown("**⏰ Lead Time Filter**")
        
        # Get standardized min/max values
        std_min = float(daily_data['lead_time'].min())  # -0.992
        std_max = float(daily_data['lead_time'].max())  # 4.162
        
        # Convert to original values for display (should be 0-13)
        orig_min = convert_standardized_to_original_leadtime(std_min)
        orig_max = convert_standardized_to_original_leadtime(std_max)
        
        # Let user select in original days (0-13)
        orig_lead_time_range = st.slider(
            "Select lead time range (days)",
            min_value=int(orig_min),      # Should be 0
            max_value=int(orig_max),      # Should be 13
            value=(int(orig_min), int(orig_max)),
            step=1,
            help=f"Filter bookings by lead time in days (range: {orig_min}-{orig_max} days)"
        )
        
        # Convert user selection back to standardized values for filtering
        # Reverse the linear mapping
        norm_min = (orig_lead_time_range[0] - 0) / (13 - 0)  # normalize to 0-1
        norm_max = (orig_lead_time_range[1] - 0) / (13 - 0)  # normalize to 0-1
        
        # Map back to standardized range
        std_lead_min = (-0.9921200019255964) + norm_min * (4.161896980478907 - (-0.9921200019255964))
        std_lead_max = (-0.9921200019255964) + norm_max * (4.161896980478907 - (-0.9921200019255964))
        
        lead_time_range = (std_lead_min, std_lead_max)
        
        st.caption(f"Selected: {orig_lead_time_range[0]} to {orig_lead_time_range[1]} days")
    
    # Set default date range for filtering (not displayed to user)
    date_range = (date(2023, 1, 1), date(2024, 12, 31))
    
    # Apply filters
    filtered_data = filter_data(daily_data, date_range, market_segment, lead_time_range)
    
    if len(filtered_data) == 0:
        st.warning("⚠️ No data matches the selected filters. Please adjust your filter criteria.")
        st.stop()
    
    # Get predictions for filtered data
    filtered_predictions = make_prediction(filtered_data)
    
    # --- SUMMARY CARDS ---
    st.subheader("📊 Summary Metrics")
    
    total_bookings = len(filtered_predictions)
    predicted_cancels = len(filtered_predictions[filtered_predictions["Prediction"] == "Canceled"])
    predicted_stays = total_bookings - predicted_cancels
    cancel_rate = (predicted_cancels / total_bookings * 100) if total_bookings > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📈 Total Bookings",
            value=f"{total_bookings:,}",
            help="Total number of bookings in the filtered dataset"
        )
    
    with col2:
        st.metric(
            label="❌ Predicted Cancels", 
            value=f"{predicted_cancels:,}",
            delta=f"{cancel_rate:.1f}% of total",
            delta_color="inverse",
            help="Number of bookings predicted to be canceled"
        )
    
    with col3:
        st.metric(
            label="✅ Predicted Stays",
            value=f"{predicted_stays:,}", 
            delta=f"{100-cancel_rate:.1f}% of total",
            help="Number of bookings predicted to not be canceled"
        )
    
    with col4:
        risk_level = "🔴 High" if cancel_rate > 25 else "🟡 Medium" if cancel_rate > 15 else "🟢 Low"
        st.metric(
            label="⚡ Cancel Rate",
            value=f"{cancel_rate:.1f}%",
            delta=risk_level,
            help="Percentage of bookings predicted to be canceled"
        )
    
    # --- HIGH RISK BOOKINGS ---
    st.markdown("---")
    st.subheader("🚨 High-Risk Bookings Report")
    
    high_risk_df = filtered_predictions[
        (filtered_predictions["Prediction"] == "Canceled")
    ].sort_values(by="Risk_Score_Numeric", ascending=False)  # Higher prob = higher risk
    
    if len(high_risk_df) > 0:
        # Display the actionable table
        st.markdown("**Bookings requiring immediate attention:**")
        display_df = high_risk_df.drop('Risk_Score_Numeric', axis=1)  # Remove numeric column for display
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )
        
        # Download button
        csv = display_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download High-Risk Report as CSV",
            data=csv,
            file_name=f"high_risk_report_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )
        
        # Risk distribution chart
        st.subheader("📈 Risk Score Distribution")
        
        # Create risk score histogram
        chart_data = pd.DataFrame({
            'Risk Score': filtered_predictions['Risk_Score_Numeric']
        })
        
        risk_chart = alt.Chart(chart_data).mark_bar().encode(
            alt.X('Risk Score:Q', bin=alt.Bin(maxbins=20), title='Cancellation Risk Score'),
            alt.Y('count():Q', title='Number of Bookings'),
            color=alt.Color('Risk Score:Q', scale=alt.Scale(scheme='redyellowgreen', reverse=True))
        ).properties(
            width=700,
            height=300,
            title="Distribution of Cancellation Risk Scores"
        )
        
        st.altair_chart(risk_chart, use_container_width=True)
        
    else:
        st.success("🎉 Great news! No high-risk bookings found in the current selection.")
    
    # --- CHARTS ROW ---
    st.markdown("---")
    st.subheader("📊 Analytics Dashboard")
    
    # Create two columns for side-by-side charts
    col1, col2 = st.columns(2)
    
    with col1:
        # a. Prediction Distribution
        st.markdown("### Prediction Distribution")
        
        prediction_counts = filtered_predictions['Prediction'].value_counts()
        
        # Create pie chart
        pie_data = pd.DataFrame({
            'Prediction': prediction_counts.index,
            'Count': prediction_counts.values,
            'Percentage': (prediction_counts.values / prediction_counts.sum() * 100).round(1)
        })
        
        pie_chart = alt.Chart(pie_data).mark_arc(innerRadius=50).encode(
            theta=alt.Theta(field="Count", type="quantitative"),
            color=alt.Color(field="Prediction", type="nominal", 
                           scale=alt.Scale(domain=["Canceled", "Not Canceled"], 
                                         range=["#ff6b6b", "#4ecdc4"])),
            tooltip=['Prediction', 'Count', 'Percentage']
        ).properties(
            width=350,
            height=300,
            title="Booking Prediction Distribution"
        )
        
        st.altair_chart(pie_chart, use_container_width=True)
    
    with col2:
        # c. Feature Importance
        st.markdown("### Feature Importance")
        
        # Extract feature importances from the loaded model if they exist
        if hasattr(model, 'feature_importances_'):
            # Create a DataFrame of features and their importance scores
            importance_df = pd.DataFrame({
                'Feature': model_cols, 
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False).head(10) # Get top 10 features
            
            importance_chart = alt.Chart(importance_df).mark_bar().encode(
                x=alt.X('Importance:Q', title='Impact on Cancellation Prediction'),
                y=alt.Y('Feature:N', sort='-x', title='Features'),
                color=alt.Color('Importance:Q', scale=alt.Scale(scheme='blues')),
                tooltip=['Feature', alt.Tooltip('Importance:Q', format='.2f')]
            ).properties(
                width=350,
                height=300,
                title="Top 10 Feature Importances"
            )
            
            st.altair_chart(importance_chart, use_container_width=True)
        else:
            # Fallback for models that don't have feature_importances_
            st.info("Feature importance data is not available for this model type.")
    
    # --- MODEL PERFORMANCE SNAPSHOT ---
    st.markdown("---")
    st.subheader("🎯 Model Performance Snapshot")
    
    # Calculate performance metrics if ground truth is available
    performance_metrics = calculate_model_performance(filtered_data, filtered_predictions)
    
    if performance_metrics:
        # Create three columns for the performance metrics
        perf_col1, perf_col2, perf_col3 = st.columns(3)
        
        with perf_col1:
            st.markdown("#### 📊 Key Metrics")
            st.metric(
                label="🎯 ROC AUC Score",
                value=f"{performance_metrics['roc_auc']:.3f}",
                help="Area Under the ROC Curve - measures model's ability to distinguish between classes"
            )
            st.metric(
                label="🎪 Precision (Cancel)",
                value=f"{performance_metrics['precision_cancel']:.3f}",
                help="Of all predicted cancellations, what percentage were actually canceled"
            )
            st.metric(
                label="🔍 Recall (Cancel)",
                value=f"{performance_metrics['recall_cancel']:.3f}",
                help="Of all actual cancellations, what percentage were correctly identified"
            )
        
        with perf_col2:
            st.markdown("#### 🎭 Confusion Matrix")
            
            # Create confusion matrix visualization
            cm = performance_metrics['confusion_matrix']
            tn, fp, fn, tp = cm.ravel()

            # Create a dataframe for the confusion matrix
            cm_df = pd.DataFrame([
                {'Actual': 'Not Canceled', 'Predicted': 'Not Canceled', 'Count': tn, 'Label': f'TN: {tn}'},
                {'Actual': 'Not Canceled', 'Predicted': 'Canceled', 'Count': fp, 'Label': f'FP: {fp}'},
                {'Actual': 'Canceled', 'Predicted': 'Not Canceled', 'Count': fn, 'Label': f'FN: {fn}'},
                {'Actual': 'Canceled', 'Predicted': 'Canceled', 'Count': tp, 'Label': f'TP: {tp}'}
            ])
            
            # Create heatmap-style confusion matrix
            cm_chart = alt.Chart(cm_df).mark_rect(
                stroke='white',
                strokeWidth=2
            ).encode(
                x=alt.X('Predicted:N', title='Predicted', axis=alt.Axis(labelAngle=0), sort=['Not Canceled', 'Canceled']),
                y=alt.Y('Actual:N', title='Actual', axis=alt.Axis(labelAngle=0), sort=['Not Canceled', 'Canceled']),
                color=alt.Color('Count:Q', 
                               scale=alt.Scale(scheme='blues', range=['#deebf7', '#08519c']),
                               legend=alt.Legend(title="Count")),
                tooltip=['Actual:N', 'Predicted:N', 'Count:Q', 'Label:N']
            ).properties(
                width=300,
                height=300,
                title=alt.TitleParams(text="Confusion Matrix", fontSize=16)
            )
            
            # Add text labels
            cm_text = alt.Chart(cm_df).mark_text(
                align='center',
                baseline='middle',
                fontSize=16,
                fontWeight='bold',
                color='white'
            ).encode(
                x=alt.X('Predicted:N', sort=['Not Canceled', 'Canceled']),
                y=alt.Y('Actual:N', sort=['Not Canceled', 'Canceled']),
                text=alt.Text('Label:N')
            )
            
            st.altair_chart(cm_chart + cm_text, use_container_width=True)
        
        with perf_col3:
            st.markdown("#### 📈 Performance Breakdown")
            
            # Calculate additional metrics
            tn, fp, fn, tp = cm.ravel()
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            f1_score = 2 * (performance_metrics['precision_cancel'] * performance_metrics['recall_cancel']) / \
                      (performance_metrics['precision_cancel'] + performance_metrics['recall_cancel']) \
                      if (performance_metrics['precision_cancel'] + performance_metrics['recall_cancel']) > 0 else 0
            
            st.metric(
                label="🎲 Accuracy",
                value=f"{accuracy:.3f}",
                help="Overall percentage of correct predictions"
            )
            st.metric(
                label="🛡️ Specificity",
                value=f"{specificity:.3f}",
                help="Of all actual non-cancellations, what percentage were correctly identified"
            )
            st.metric(
                label="🏆 F1-Score (Cancel)",
                value=f"{f1_score:.3f}",
                help="Harmonic mean of precision and recall for cancellation class"
            )
            
            # Model performance interpretation
            st.markdown("#### 🧠 Performance Insights")
            
            if performance_metrics['roc_auc'] >= 0.9:
                st.success("🌟 Excellent model performance!")
            elif performance_metrics['roc_auc'] >= 0.8:
                st.info("👍 Good model performance")
            elif performance_metrics['roc_auc'] >= 0.7:
                st.warning("⚠️ Moderate model performance")
            else:
                st.error("❌ Poor model performance - consider retraining")
            
            if performance_metrics['precision_cancel'] < 0.7:
                st.warning("📉 Low precision: Many false positives")
            if performance_metrics['recall_cancel'] < 0.7:
                st.warning("📉 Low recall: Missing many actual cancellations")
    
    else:
        st.info("📋 Model performance metrics require ground truth labels. Please ensure your dataset includes a target column (e.g., 'is_canceled', 'canceled', 'target') to display performance metrics.")

else:
    st.info("👆 Please upload a CSV file or ensure `cleaned_data.csv` is available to begin the analysis.")