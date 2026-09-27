import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

from src.data_loader import load_cmapss_data
from src.feature_engineering import engineer_features
from src.preprocessing import load_scaler, scale_data
from src.predict import predict_rul
from src.maintenance import get_maintenance_decision


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Predictive Maintenance AI",
    page_icon="🔧",
    layout="wide"
)


# --------------------------------------------------
# Session state initialization
# --------------------------------------------------

if "results_df" not in st.session_state:
    st.session_state.results_df = None


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🔧 Predictive Maintenance AI")

st.write(
    """
    This application uses a stacked LSTM model to estimate
    the Remaining Useful Life (RUL) of aircraft engines
    and generate maintenance recommendations.
    """
)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.header("Model Information")

st.sidebar.write(
    "Model: Stacked LSTM"
)

st.sidebar.write(
    "Sequence length: 30 cycles"
)

st.sidebar.write(
    "Input features: 112"
)


# --------------------------------------------------
# File upload
# --------------------------------------------------

st.header("Upload Engine Data")

uploaded_file = st.file_uploader(
    "Upload a C-MAPSS sensor data file",
    type=["txt"]
)


# --------------------------------------------------
# Run prediction
# --------------------------------------------------

if uploaded_file is not None:

    st.success(
        "File uploaded successfully."
    )

    # Load uploaded file
    test_df = load_cmapss_data(
        uploaded_file
    )

    st.subheader("Dataset Information")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Number of engines",
            test_df["unit"].nunique()
        )

    with col2:
        st.metric(
            "Number of observations",
            len(test_df)
        )

    if st.button(
        "Run Predictive Maintenance Analysis"
    ):

        with st.spinner(
            "Running the predictive maintenance model..."
        ):

            # ------------------------------------------
            # Feature engineering
            # ------------------------------------------

            test_features, _ = engineer_features(
                test_df,
                calculate_target=False
            )

            # ------------------------------------------
            # Scaling
            # ------------------------------------------

            scaler = load_scaler()

            test_scaled = scale_data(
                test_features,
                scaler
            )

            # ------------------------------------------
            # RUL prediction
            # ------------------------------------------

            predictions = predict_rul(
                test_scaled
            )

            # ------------------------------------------
            # Maintenance decisions
            # ------------------------------------------

            results = []

            for unit_id, predicted_rul in predictions.items():

                decision = get_maintenance_decision(
                    predicted_rul
                )

                results.append({
                    "Engine": unit_id,
                    "Predicted RUL": round(
                        predicted_rul,
                        2
                    ),
                    "Status": decision["status"],
                    "Recommended Action": decision["action"]
                })

            results_df = pd.DataFrame(
                results
            )

            results_df = results_df.sort_values(
                "Engine"
            ).reset_index(drop=True)

            # ------------------------------------------
            # SAVE RESULTS IN SESSION STATE
            # ------------------------------------------

            st.session_state.results_df = results_df


# --------------------------------------------------
# Display results if prediction exists
# --------------------------------------------------

if st.session_state.results_df is not None:

    results_df = st.session_state.results_df


    # ----------------------------------------------
    # Maintenance Overview
    # ----------------------------------------------

    st.header("Maintenance Overview")

    normal_count = (
        results_df["Status"] == "NORMAL"
    ).sum()

    warning_count = (
        results_df["Status"] == "WARNING"
    ).sum()

    critical_count = (
        results_df["Status"] == "CRITICAL"
    ).sum()


    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "🟢 Normal",
            normal_count
        )

    with col2:
        st.metric(
            "🟡 Warning",
            warning_count
        )

    with col3:
        st.metric(
            "🔴 Critical",
            critical_count
        )


    # ----------------------------------------------
    # Results table
    # ----------------------------------------------

    st.header("Engine Predictions")

    st.dataframe(
        results_df,
        use_container_width=True
    )


    # ----------------------------------------------
    # Download results
    # ----------------------------------------------

    csv = results_df.to_csv(
        index=False
    )

    st.download_button(
        label="Download Predictions",
        data=csv,
        file_name="maintenance_predictions.csv",
        mime="text/csv"
    )


    # ----------------------------------------------
    # Outlook Maintenance Scheduling
    # ----------------------------------------------

    st.header("📅 Maintenance Scheduling")

    critical_engines = results_df[
        results_df["Status"] == "CRITICAL"
    ].copy()


    if len(critical_engines) == 0:

        st.success(
            "No critical engines require immediate maintenance."
        )

    else:

        st.warning(
            f"{len(critical_engines)} engine(s) require maintenance."
        )


        # ------------------------------------------
        # Select engine
        # ------------------------------------------

        selected_engine = st.selectbox(
            "Select an engine to schedule maintenance",
            critical_engines["Engine"].tolist()
        )


        selected_row = critical_engines[
            critical_engines["Engine"] == selected_engine
        ].iloc[0]


        st.write(
            f"**Predicted RUL:** "
            f"{selected_row['Predicted RUL']} cycles"
        )

        st.write(
            f"**Status:** "
            f"{selected_row['Status']}"
        )

        st.write(
            f"**Recommended action:** "
            f"{selected_row['Recommended Action']}"
        )


        # ------------------------------------------
        # Maintenance date/time
        # ------------------------------------------

        st.subheader("Schedule Maintenance")


        maintenance_date = st.date_input(
            "Maintenance date"
        )


        maintenance_time = st.time_input(
            "Maintenance time"
        )


        duration_minutes = st.number_input(
            "Maintenance duration (minutes)",
            min_value=15,
            max_value=480,
            value=60,
            step=15
        )


        # ------------------------------------------
        # Schedule button
        # ------------------------------------------

        if st.button(
            "📅 Schedule Maintenance in Outlook",
            type="primary"
        ):

            start_datetime = datetime.combine(
                maintenance_date,
                maintenance_time
            )

            end_datetime = (
                start_datetime
                + timedelta(minutes=duration_minutes)
            )

            event_subject = (
                f"Aircraft Engine {selected_engine} "
                f"Maintenance"
            )

            event_body = (
                f"Predictive Maintenance Recommendation\n\n"
                f"Engine: {selected_engine}\n"
                f"Predicted RUL: "
                f"{selected_row['Predicted RUL']} cycles\n"
                f"Status: {selected_row['Status']}\n"
                f"Recommended Action: "
                f"{selected_row['Recommended Action']}\n\n"
                f"Scheduled automatically by "
                f"Predictive Maintenance AI."
            )

            # We will connect Outlook here later.
            st.success(
                "Maintenance event is ready to be scheduled."
            )