import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import matplotlib.pyplot as plt
from datetime import datetime

# Import utility functions
from utils.s3_utils import (
    upload_to_s3, 
    download_from_s3, 
    list_s3_files,
    generate_presigned_url
)
from utils.prophet_utils import (
    fit_prophet_model,
    make_forecast,
    cross_validate_model,
    get_performance_metrics,
    optimize_hyperparameters,
    save_model_to_json,
    load_model_from_json
)

# Set page configuration
st.set_page_config(
    page_title="AWS-Powered Forecast App",
    initial_sidebar_state="expanded",
    page_icon="🔮"
)

# Configure environment
# Default bucket name - change this according to your setup
S3_BUCKET = os.environ.get('S3_BUCKET_NAME', 'forecast-app-bucket')

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'forecast' not in st.session_state:
    st.session_state.forecast = None
if 'df' not in st.session_state:
    st.session_state.df = None

# App navigation
tabs = ["Data", "Forecast", "Validation", "AWS Operations"]
selected_tab = st.sidebar.selectbox("Navigation", tabs)

def load_csv(file_buffer):
    """Load a CSV file into a pandas DataFrame"""
    try:
        df_input = pd.read_csv(file_buffer, sep=None, engine='python', encoding='utf-8',
                            parse_dates=True, infer_datetime_format=True)
        return df_input
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return None

def prep_data(df, date_col, metric_col):
    """Prepare data for Prophet"""
    try:
        df_input = df.rename({date_col: "ds", metric_col: "y"}, errors='raise', axis=1)
        df_input = df_input[['ds', 'y']]
        df_input = df_input.sort_values(by='ds', ascending=True)
        return df_input
    except Exception as e:
        st.error(f"Error preparing data: {e}")
        return None

# DATA TAB
if selected_tab == "Data":
    st.title('Data Upload and Preparation 📊')
    
    data_source = st.radio(
        "Select data source",
        ["Upload a file", "Load from S3"]
    )
    
    if data_source == "Upload a file":
        uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
        
        if uploaded_file is not None:
            # Save to S3 first
            file_key = f"uploads/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uploaded_file.name}"
            
            with st.spinner('Uploading to S3...'):
                upload_success = upload_to_s3(uploaded_file, S3_BUCKET, file_key)
                
                if upload_success:
                    st.success(f"File uploaded to S3: {file_key}")
                    # Reset uploaded_file to position 0
                    uploaded_file.seek(0)
                    # Load the data
                    df = load_csv(uploaded_file)
                    
                    if df is not None:
                        st.session_state.raw_df = df
                        st.write("Data preview:")
                        st.dataframe(df.head())
    
    elif data_source == "Load from S3":
        with st.spinner('Listing S3 files...'):
            files = list_s3_files(S3_BUCKET, prefix="uploads/")
            
        if not files:
            st.info("No files found in S3 bucket. Upload a file first.")
        else:
            selected_file = st.selectbox("Select a file from S3", options=files)
            
            if st.button("Load selected file"):
                with st.spinner('Loading from S3...'):
                    file_data = download_from_s3(S3_BUCKET, selected_file)
                    
                    if file_data:
                        df = load_csv(file_data)
                        if df is not None:
                            st.session_state.raw_df = df
                            st.write("Data preview:")
                            st.dataframe(df.head())
    
    # Data preparation section
    if 'raw_df' in st.session_state and st.session_state.raw_df is not None:
        st.subheader("Data Preparation")
        
        columns = list(st.session_state.raw_df.columns)
        
        col1, col2 = st.columns(2)
        with col1:
            date_col = st.selectbox("Select date column", options=columns, index=0)
        with col2:
            metric_col = st.selectbox("Select value column", options=columns, index=1 if len(columns) > 1 else 0)
        
        if st.button("Prepare Data for Forecasting"):
            prepared_df = prep_data(st.session_state.raw_df, date_col, metric_col)
            
            if prepared_df is not None:
                st.session_state.df = prepared_df
                st.success("Data prepared successfully!")
                
                st.write("Prepared data preview:")
                st.dataframe(prepared_df.head())
                
                # Show a preview chart
                st.subheader("Time Series Preview")
                chart_data = prepared_df.rename(columns={"ds": "date", "y": "value"})
                st.line_chart(chart_data, x="date", y="value")

# FORECAST TAB
elif selected_tab == "Forecast":
    st.title('Time Series Forecasting 🔮')
    
    if st.session_state.df is None:
        st.warning("Please load and prepare data in the Data tab first.")
    else:
        st.write('Configure your forecasting model parameters.')
        
        st.subheader("Model Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            periods_input = st.number_input(
                'Forecast horizon (days)',
                min_value=1, 
                max_value=366,
                value=90
            )
            
            seasonality = st.radio(
                "Seasonality mode",
                options=['additive', 'multiplicative']
            )
            
            growth = st.radio(
                "Growth model",
                options=['linear', 'logistic']
            )
            
        with col2:
            daily = st.checkbox("Daily seasonality")
            weekly = st.checkbox("Weekly seasonality", value=True)
            monthly = st.checkbox("Monthly seasonality")
            yearly = st.checkbox("Yearly seasonality", value=True)
            
            if growth == 'logistic':
                cap = st.slider('Cap value', min_value=0.0, max_value=10.0, step=0.1, value=1.0)
                floor = st.slider('Floor value', min_value=0.0, max_value=1.0, step=0.1, value=0.0)
            
        with st.expander("Advanced Parameters"):
            changepoint_scale = st.slider(
                'Changepoint prior scale',
                min_value=0.001, 
                max_value=0.5, 
                value=0.05,
                step=0.001,
                format="%.3f"
            )
            
            seasonality_scale = st.slider(
                'Seasonality prior scale',
                min_value=0.1, 
                max_value=10.0, 
                value=10.0,
                step=0.1
            )
            
            countries = ['None', 'US', 'CA', 'UK', 'DE', 'FR', 'ES', 'IT']
            selected_country = st.selectbox("Add country holidays", options=countries)
        
        # Model fitting button
        if st.button("Fit Model"):
            with st.spinner('Training the model...'):
                # Prepare parameters dictionary
                params = {
                    'seasonality_mode': seasonality,
                    'daily_seasonality': daily,
                    'weekly_seasonality': weekly,
                    'yearly_seasonality': yearly,
                    'growth': growth,
                    'changepoint_prior_scale': changepoint_scale,
                    'seasonality_prior_scale': seasonality_scale
                }
                
                # Add holidays if selected
                if selected_country != 'None':
                    params['holidays_country'] = selected_country
                
                # Add monthly seasonality if selected
                if monthly:
                    params['monthly_seasonality'] = True
                
                # For logistic growth, add cap and floor to dataframe
                if growth == 'logistic':
                    df_copy = st.session_state.df.copy()
                    df_copy['cap'] = cap
                    df_copy['floor'] = floor
                else:
                    df_copy = st.session_state.df
                
                # Fit the model
                st.session_state.model = fit_prophet_model(df_copy, params)
                
                if st.session_state.model:
                    st.success("Model trained successfully!")
                    
                    # Save model to S3
                    model_json = save_model_to_json(st.session_state.model)
                    if model_json:
                        model_key = f"models/prophet_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                        model_bytes = io.BytesIO(model_json.encode('utf-8'))
                        
                        upload_success = upload_to_s3(
                            model_bytes, 
                            S3_BUCKET, 
                            model_key,
                            content_type='application/json'
                        )
                        
                        if upload_success:
                            st.success(f"Model saved to S3: {model_key}")
        
        # Generate forecast if model exists
        if st.session_state.model is not None:
            if st.button("Generate Forecast"):
                with st.spinner('Generating forecast...'):
                    # For logistic growth, set cap and floor
                    if st.session_state.model.growth == 'logistic':
                        cap_val = cap
                        floor_val = floor
                    else:
                        cap_val = None
                        floor_val = None
                    
                    # Generate forecast
                    st.session_state.forecast = make_forecast(
                        st.session_state.model,
                        periods=periods_input
                    )
                    
                    if st.session_state.forecast is not None:
                        st.success("Forecast generated successfully!")
                        
                        # Save forecast to S3
                        forecast_key = f"forecasts/forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                        forecast_csv = st.session_state.forecast.to_csv(index=False).encode('utf-8')
                        
                        upload_success = upload_to_s3(
                            io.BytesIO(forecast_csv), 
                            S3_BUCKET, 
                            forecast_key,
                            content_type='text/csv'
                        )
                        
                        if upload_success:
                            st.success(f"Forecast saved to S3: {forecast_key}")
                            # Create a download link
                            url = generate_presigned_url(S3_BUCKET, forecast_key)
                            if url:
                                st.markdown(f"[Download Forecast CSV]({url})")
            
            # Show forecast results if available
            if st.session_state.forecast is not None:
                st.subheader("Forecast Results")
                
                # Show forecast table
                if st.checkbox("Show forecast table"):
                    st.dataframe(st.session_state.forecast)
                
                # Show forecast plot
                st.subheader("Forecast Plot")
                
                fig = plt.figure(figsize=(10, 6))
                
                # Get the model and forecast from session state
                m = st.session_state.model
                forecast = st.session_state.forecast
                
                # Create the plot
                ax = fig.add_subplot(111)
                m.plot(forecast, ax=ax)
                
                st.pyplot(fig)
                
                # Show components plot
                if st.checkbox("Show components"):
                    fig2 = plt.figure(figsize=(10, 8))
                    m.plot_components(forecast, fig=fig2)
                    st.pyplot(fig2)

# VALIDATION TAB
elif selected_tab == "Validation":
    st.title('Model Validation and Tuning 🧪')
    
    if st.session_state.model is None:
        st.warning("Please train a model in the Forecast tab first.")
    else:
        st.write("Validate and optimize your forecasting model.")
        
        st.subheader("Cross Validation")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            initial = st.text_input("Initial training period", value="365 days")
        
        with col2:
            period = st.text_input("Period between cutoffs", value="30 days")
        
        with col3:
            horizon = st.text_input("Forecast horizon", value="90 days")
        
        if st.button("Run Cross Validation"):
            with st.spinner("Running cross validation..."):
                cv_results = cross_validate_model(
                    st.session_state.model,
                    initial=initial,
                    period=period,
                    horizon=horizon
                )
                
                if cv_results is not None:
                    st.session_state.cv_results = cv_results
                    st.success("Cross validation completed!")
                    
                    # Calculate performance metrics
                    performance = get_performance_metrics(cv_results)
                    
                    if performance is not None:
                        st.session_state.performance = performance
                        
                        st.subheader("Performance Metrics")
                        st.dataframe(performance)
                        
                        # Save metrics to S3
                        metrics_key = f"metrics/metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                        metrics_csv = performance.to_csv(index=False).encode('utf-8')
                        
                        upload_success = upload_to_s3(
                            io.BytesIO(metrics_csv), 
                            S3_BUCKET, 
                            metrics_key,
                            content_type='text/csv'
                        )
                        
                        if upload_success:
                            st.success(f"Metrics saved to S3: {metrics_key}")
        
        # Hyperparameter tuning section
        st.subheader("Hyperparameter Tuning")
        
        if st.button("Run Hyperparameter Tuning"):
            with st.spinner("This may take a while..."):
                # Define parameter grid
                param_grid = {  
                    'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
                    'seasonality_prior_scale': [0.1, 1.0, 10.0],
                }
                
                # Run optimization
                tuning_results = optimize_hyperparameters(
                    st.session_state.df,
                    param_grid=param_grid,
                    initial=initial,
                    period=period,
                    horizon=horizon
                )
                
                if tuning_results:
                    st.session_state.tuning_results = tuning_results
                    
                    st.success("Hyperparameter tuning completed!")
                    
                    st.subheader("Best Parameters")
                    st.json(tuning_results['best_params'])
                    
                    st.subheader("All Results")
                    st.dataframe(tuning_results['all_results'])
                    
                    # Save results to S3
                    tuning_key = f"tuning/tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    tuning_csv = tuning_results['all_results'].to_csv(index=False).encode('utf-8')
                    
                    upload_success = upload_to_s3(
                        io.BytesIO(tuning_csv), 
                        S3_BUCKET, 
                        tuning_key,
                        content_type='text/csv'
                    )
                    
                    if upload_success:
                        st.success(f"Tuning results saved to S3: {tuning_key}")

# AWS OPERATIONS TAB
elif selected_tab == "AWS Operations":
    st.title('AWS Operations 🚀')
    
    st.subheader("S3 File Management")
    
    operation = st.radio(
        "Select operation",
        ["List files", "Download file"]
    )
    
    if operation == "List files":
        prefix_options = ["uploads/", "models/", "forecasts/", "metrics/", "tuning/"]
        selected_prefix = st.selectbox("Select folder", options=prefix_options)
        
        if st.button("List files"):
            with st.spinner('Listing files...'):
                files = list_s3_files(S3_BUCKET, prefix=selected_prefix)
                
                if not files:
                    st.info(f"No files found in {selected_prefix}")
                else:
                    st.success(f"Found {len(files)} files")
                    for file in files:
                        st.write(file)
    
    elif operation == "Download file":
        # List all files in S3
        with st.spinner('Listing files...'):
            files = list_s3_files(S3_BUCKET)
            
        if not files:
            st.info("No files found in S3 bucket")
        else:
            selected_file = st.selectbox("Select file to download", options=files)
            
            if st.button("Generate Download Link"):
                with st.spinner('Generating link...'):
                    url = generate_presigned_url(S3_BUCKET, selected_file)
                    
                    if url:
                        st.success("Download link generated!")
                        st.markdown(f"[Download {selected_file}]({url})")
                    else:
                        st.error("Error generating download link")

    # Add bucket information
    st.sidebar.subheader("S3 Bucket Information")
    st.sidebar.info(f"Current bucket: {S3_BUCKET}")
    st.sidebar.write("To change the bucket, set the S3_BUCKET_NAME environment variable")

if __name__ == "__main__":
    # Add about information in the sidebar
    st.sidebar.title("About")
    st.sidebar.info(
        "This app demonstrates AWS integration with a "
        "time series forecasting application built with Streamlit and Prophet."
    )
    st.sidebar.write("Version: 1.0.0")