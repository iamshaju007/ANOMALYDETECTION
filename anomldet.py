import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import streamlit as st

# Function to load data from multiple CSV files
def load_data(csv_files):
    data_frames = []
    for file in csv_files:
        df = pd.read_csv(file, encoding='ISO-8859-1')
        df = df.rename(columns=lambda x: x.strip())  # Remove extra spaces
        data_frames.append(df)
    
    data = pd.concat(data_frames, ignore_index=True)
    
    if 'timestamp' in data.columns and 'value' in data.columns:
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data = data[['timestamp', 'value']]  # Keep only the relevant columns
        data = data.fillna(method='ffill')
        return data
    else:
        raise KeyError("The required columns ('timestamp' and 'value') are missing from the data.")

# Function to perform anomaly detection with training and testing
def perform_anomaly_detection(data, contamination_level=0.01):
    # Prepare features and target
    features = ['value']
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data[features])

    # Initialize and fit the model
    model = IsolationForest(contamination=contamination_level, random_state=42)
    model.fit(data_scaled)

    # Predict anomalies
    data['anomaly'] = model.predict(data_scaled)
    data['anomaly'] = data['anomaly'].map({1: 0, -1: 1})  # Map anomalies to 1 for clarity

    # Calculate performance metrics
    metrics = {}
    if data['anomaly'].sum() > 0:
        true_labels = [0] * len(data)  # Modify if true labels are available
        predicted_labels = data['anomaly']
        metrics = classification_report(true_labels, predicted_labels, output_dict=True, zero_division=0)

    return data, metrics

# Function to generate conclusions based on anomaly detection
def generate_conclusion(data):
    anomaly_count = data['anomaly'].sum()
    total_points = len(data)

    conclusion = f"In the dataset, {anomaly_count} anomalies were detected out of {total_points} data points.\n\n"

    if anomaly_count > 0:
        conclusion += ("Anomalies are deviations from the expected pattern, which may represent unusual behavior or outliers in the data. "
                       "These could signal underlying problems, such as sudden spikes in usage, system failures, or irregular operational behaviors.\n\n"
                       "A further investigation should be performed to assess the root cause of these anomalies, which could involve looking at the time "
                       "range of the anomalies to correlate with specific events or incidents.\n")
    else:
        conclusion += "No significant anomalies were detected in the dataset, indicating that the data trends are normal and consistent over time.\n\n"

    return conclusion

# Function to create Streamlit app
def streamlit_app():
    st.title('Anomaly Detection in Time Series Data')

    # File uploader for multiple CSV files
    uploaded_files = st.file_uploader("Choose one or more CSV files", type="csv", accept_multiple_files=True)
    if uploaded_files:
        try:
            # Load data from uploaded CSV files
            data = load_data(uploaded_files)

            # Display the entire dataset
            st.write("### Full Data:")
            st.dataframe(data, use_container_width=True)  # Display full dataset with scrollable table

            # Display data sample
            st.write("### Data Sample (First 5 rows):")
            st.write(data.head())

            # Date range selection
            st.write("### Select Time Range:")
            min_date = data['timestamp'].min()
            max_date = data['timestamp'].max()
            start_date = st.date_input('Start date', min_value=min_date.date(), max_value=max_date.date(), value=min_date.date())
            end_date = st.date_input('End date', min_value=min_date.date(), max_value=max_date.date(), value=max_date.date())

            if start_date > end_date:
                st.error("Error: End date must fall after start date.")
                return

            # Filter data based on selected date range
            mask = (data['timestamp'] >= pd.to_datetime(start_date)) & (data['timestamp'] <= pd.to_datetime(end_date))
            data = data.loc[mask]
            data.set_index('timestamp', inplace=True)

            # Contamination threshold slider
            contamination_level = st.slider('Anomaly Detection Sensitivity (contamination level)', min_value=0.01, max_value=0.1, value=0.01, step=0.01)

            # Perform anomaly detection with slider-controlled contamination level
            data, metrics = perform_anomaly_detection(data, contamination_level)

            # Display summary statistics
            st.write("### Summary Statistics:")
            st.write({
                'Mean': data['value'].mean(),
                'Median': data['value'].median(),
                'Standard Deviation': data['value'].std()
            })

            # Data Analysis
            st.write("### Data Analysis:")
            st.write(f"Total data points: {len(data)}")
            st.write(f"Time Range: {data.index.min()} to {data.index.max()}")

            # Graph - Anomaly Detection Results
            st.write("### Anomaly Detection Results:")

            # Improved Matplotlib Plot
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(data.index, data['value'], label='Value', color='blue', alpha=0.6, lw=2)

            # Highlight anomalies
            anomalies = data[data['anomaly'] == 1]
            ax.scatter(anomalies.index, anomalies['value'], color='red', label='Anomalies', s=50, marker='x')

            ax.set_title('Anomaly Detection in Time Series Data', fontsize=16)
            ax.set_xlabel('Timestamp', fontsize=14)
            ax.set_ylabel('Value', fontsize=14)
            ax.legend()
            st.pyplot(fig)

            # Interactive Plot using Plotly
            st.write("#### Interactive Plot:")
            data_reset = data.reset_index()
            fig = px.scatter(data_reset, x='timestamp', y='value', color='anomaly', title='Anomaly Detection',
                             labels={'anomaly': 'Anomaly Status'}, color_discrete_map={0: 'blue', 1: 'red'},
                             hover_data={'timestamp': True, 'value': True})

            # Customize marker size and border
            fig.update_traces(marker=dict(size=8, line=dict(width=1, color='DarkSlateGrey')))

            # Update layout for better visualization
            fig.update_layout(
                legend_title_text='Anomaly',
                xaxis_title='Timestamp',
                yaxis_title='Value',
                xaxis_rangeslider_visible=True  # Add zoom slider
            )

            # Display the Plotly chart
            st.plotly_chart(fig)

            # Model Performance Metrics
            st.write("### Model Performance Metrics:")
            if metrics:
                st.write(f"Precision: {metrics['1']['precision']:.2f}")
                st.write(f"Recall: {metrics['1']['recall']:.2f}")
                st.write(f"F1 Score: {metrics['1']['f1-score']:.2f}")
            else:
                st.write("No meaningful metrics available for the detected anomalies.")

            # Conclusion
            st.write("### Conclusion:")
            conclusion = generate_conclusion(data)
            st.write(conclusion)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    streamlit_app()

