import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --- Model Training (Caching) ---
# This function now loads REAL data, trains the model, and calculates its metrics.
@st.cache_resource
def train_model():
    """
    Loads real data from 'student_exam_scores.csv', trains the model
    as described in the user's code, and returns the trained model
    and its performance metrics.
    """
    # 1. Load the real dataset
    try:
        data = pd.read_csv('student_exam_scores.csv')
    except FileNotFoundError:
        st.error("Error: The file 'student_exam_scores.csv' was not found.")
        st.info("Please make sure the CSV file is in the same directory as your app.py script.")
        st.stop()  # Stop the app from running further
    except Exception as e:
        st.error(f"An error occurred while loading the data: {e}")
        st.stop()

    # 2. Replicate the user's model training pipeline
    try:
        X_multi = data[['hours_studied', 'sleep_hours', 'attendance_percent', 'previous_scores']]
        y_multi = data['exam_score']
    except KeyError as e:
        st.error(f"Error: Your CSV file is missing an expected column: {e}.")
        st.info(
            "Please ensure your CSV has: 'hours_studied', 'sleep_hours', 'attendance_percent', 'previous_scores', and 'exam_score'.")
        st.stop()

    # 3. Splitting the dataset
    X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
        X_multi, y_multi, test_size=0.2, random_state=42
    )

    # 4. Initializing and training the model
    model_multi = LinearRegression()
    model_multi.fit(X_train_m, y_train_m)

    # 5. Evaluate the model on the test set to get TRUE metrics
    y_pred_m = model_multi.predict(X_test_m)
    mae = mean_absolute_error(y_test_m, y_pred_m)
    mse = mean_squared_error(y_test_m, y_pred_m)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_m, y_pred_m)

    # Store metrics in a dictionary to return
    metrics_dict = {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2
    }

    return model_multi, metrics_dict


# --- Load the Model and Metrics ---
# This line calls the function above and caches the results
model, metrics = train_model()

# --- Streamlit App UI ---

# Page configuration
st.set_page_config(layout="wide", page_title="Exam Score Predictor")

# Title and description
st.title("🎓 Student Exam Score Predictor")
st.write("""
This app uses a **Multiple Linear Regression** model to predict your final exam score. 
Provide your information using the sliders below, and the model will estimate your performance.
""")

st.divider()

# --- Input Section ---
st.subheader("Enter Your Preparation Details")

# Use columns for a cleaner layout
col1, col2 = st.columns(2)

with col1:
    # `hours_studied` (float)
    hours_studied = st.slider(
        "⏰ Hours Studied per Week",
        min_value=0.0,
        max_value=20.0,
        value=8.0,
        step=0.5
    )

    # `sleep_hours` (float)
    sleep_hours = st.slider(
        "😴 Average Sleep Hours per Night",
        min_value=0.0,
        max_value=12.0,
        value=7.0,
        step=0.5
    )

with col2:
    # `attendance_percent` (float)
    attendance_percent = st.slider(
        "📊 Class Attendance Percentage",
        min_value=0.0,
        max_value=100.0,
        value=85.0,
        step=1.0
    )

    # `previous_scores` (int)
    previous_scores = st.slider(
        "📈 Average Previous Score",
        min_value=0,
        max_value=100,
        value=70,
        step=1
    )

st.divider()

# --- Prediction and Output ---
if st.button("Predict My Score!", type="primary", use_container_width=True):
    # 1. Create a DataFrame from the inputs
    input_data = pd.DataFrame({
        'hours_studied': [hours_studied],
        'sleep_hours': [sleep_hours],
        'attendance_percent': [attendance_percent],
        'previous_scores': [previous_scores]
    })

    # 2. Make a prediction
    prediction = model.predict(input_data)
    predicted_score = prediction[0]

    # 3. Clean and display the prediction
    # Ensure the score is within the logical 0-100 range
    predicted_score_clamped = max(0, min(100, predicted_score))

    st.subheader("Your Predicted Exam Score")

    # Use st.metric for a modern, clear display
    st.metric(
        label="Estimated Score",
        value=f"{predicted_score_clamped:.1f}",
        help="This score is an estimate based on the provided inputs."
    )

    # Provide contextual feedback
    if predicted_score_clamped > 90:
        st.success("Excellent! You're on track for an top-tier score. Keep it up!")
    elif predicted_score_clamped > 75:
        st.info("Great work! You're in a strong position for a good grade.")
    elif predicted_score_clamped > 60:
        st.warning("Solid effort, but a little more study or sleep could boost your score.")
    else:
        st.error("There's significant room for improvement. Try to increase your study time and attendance.")

# --- Model Performance Expander ---
with st.expander("About This Model's Performance"):
    st.write("""
    This prediction comes from a `LinearRegression` model trained on your `student_exam_scores.csv` file.
    The model's performance was evaluated on an unseen test set (20% of the data). 
    The metrics below show how well the model performed on that test data.
    """)

    # Display the DYNAMIC metrics calculated from the REAL data
    st.subheader("Model Evaluation Metrics")

    met_col1, met_col2, met_col3 = st.columns(3)

    met_col1.metric(
        "R-squared (R²)",
        f"{metrics['r2']:.3f}",
        help="This means the model explains this percentage of the variability in exam scores."
    )
    met_col2.metric(
        "Mean Absolute Error (MAE)",
        f"{metrics['mae']:.2f}",
        help="On average, the model's prediction is off by this many points."
    )
    met_col3.metric(
        "Root Mean Squared Error (RMSE)",
        f"{metrics['rmse']:.2f}",
        help="This is another measure of error, which penalizes larger errors more heavily."
    )

    # Display model features
    st.write(f"**Model Features:** {list(model.feature_names_in_)}")