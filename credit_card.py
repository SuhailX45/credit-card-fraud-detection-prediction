import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import mysql.connector
from mysql.connector import Error

# Set page configuration
st.set_page_config(page_title="Credit Card Fraud Prediction", layout="wide")

# --- Blocked Cards ---
BLOCKED_CARDS = {"4532015112830366", "6011111111111117", "4005550000000001"}

# --- Database Functions ---
def check_login(username, password):
    """Check if username and password exist in the database."""
    try:
        connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="suhail1100",  # Replace with your actual MySQL password
            database="login"
        )
        cursor = connection.cursor()
        query = "SELECT * FROM credit_card_data WHERE username = %s AND password = %s"
        cursor.execute(query, (username, password))
        result = cursor.fetchone()
        cursor.close()
        connection.close()
        return result is not None
    except Error as e:
        st.error(f"Database connection failed: {e}")
        return False

def fetch_transactions(card_number):
    """Fetch previous transactions for a given credit card number."""
    try:
        connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="suhail1100",
            database="login"
        )
        cursor = connection.cursor()
        query = "SELECT id, credit_card_number, amount, time, location, merchant, is_fraud FROM transactions WHERE credit_card_number = %s"
        cursor.execute(query, (card_number,))
        result = cursor.fetchall()
        cursor.close()
        connection.close()
        if result:
            columns = ['ID', 'Credit Card Number', 'Amount', 'Time', 'Location', 'Merchant', 'Is Fraud']
            return pd.DataFrame(result, columns=columns)
        return pd.DataFrame(columns=['ID', 'Credit Card Number', 'Amount', 'Time', 'Location', 'Merchant', 'Is Fraud'])
    except Error as e:
        st.error(f"Error fetching transactions: {e}")
        return pd.DataFrame(columns=['ID', 'Credit Card Number', 'Amount', 'Time', 'Location', 'Merchant', 'Is Fraud'])

# --- Data Preprocessing ---
def load_data(file):
    """Load and return the dataset."""
    try:
        df = pd.read_csv(file)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def preprocess_data(df, scaler=None):
    """Preprocess the Kaggle dataset with V1-V28, Time, Amount, and Class."""
    if scaler is None:
        scaler = StandardScaler()

    df.fillna(0, inplace=True)

    # Create time-based features
    if 'Time' in df.columns:
        df['hour'] = (df['Time'] // 3600) % 24
        df['day_of_week'] = (df['Time'] // (3600 * 24)) % 7
    else:
        df['hour'] = 0
        df['day_of_week'] = 0

    # Numerical columns: V1-V28, Amount, hour, day_of_week
    numerical_cols = [f'V{i}' for i in range(1, 29)] + ['Amount', 'hour', 'day_of_week']
    available_numerical = [col for col in numerical_cols if col in df.columns]
    
    if available_numerical:
        df[available_numerical] = scaler.fit_transform(df[available_numerical])

    return df, scaler

def prepare_features(df):
    """Prepare features for model training (V1-V28, Amount, hour, day_of_week)."""
    feature_cols = [f'V{i}' for i in range(1, 29)] + ['Amount', 'hour', 'day_of_week']
    available_cols = [col for col in feature_cols if col in df.columns]
    if not available_cols:
        raise ValueError("No valid feature columns found in the dataset.")
    return df[available_cols]

# --- Model Training ---
def train_model(X, y):
    """Train a binary classification model."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBClassifier(random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, 'model.pkl')
    return model, X_test, y_test

def evaluate_model(model, X_test, y_test):
    """Evaluate the model."""
    accuracy = model.score(X_test, y_test)
    return accuracy

# --- Anomaly Detection ---
def detect_anomalies(X):
    """Detect anomalies in the data."""
    iso_forest = IsolationForest(contamination=0.01, random_state=42)
    anomalies = iso_forest.fit_predict(X)
    return anomalies

# --- Visualizations ---
def plot_transaction_distribution(df):
    """Plot distribution of transaction amounts."""
    if 'Amount' in df.columns:
        fig = px.histogram(df, x='Amount', nbins=50, title='Transaction Amount Distribution')
        st.plotly_chart(fig)

def plot_anomaly_scatter(df, anomalies):
    """Plot scatter of transactions with anomalies highlighted."""
    if 'Time' in df.columns and 'Amount' in df.columns:
        df = df.copy()
        df['anomaly'] = anomalies
        fig = px.scatter(df, x='Time', y='Amount', color='anomaly', title='Transactions with Anomalies')
        st.plotly_chart(fig)

def plot_feature_importance(model, feature_cols):
    """Plot feature importance for the trained model."""
    importances = model.feature_importances_
    fig, ax = plt.subplots()
    sns.barplot(x=importances, y=feature_cols, ax=ax)
    ax.set_title('Feature Importance')
    plt.tight_layout()
    st.pyplot(fig)

# --- Credit Card Validation ---
def luhn_check(card_number):
    """Validate a credit card number using the Luhn algorithm."""
    card_number = card_number.replace(" ", "")
    if not card_number.isdigit():
        return False
    total = 0
    reverse_digits = card_number[::-1]
    for i, digit in enumerate(reverse_digits):
        n = int(digit)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0

def credit_card_validation_page():
    st.header("Credit Card Validation")
    st.markdown("Enter a credit card number to validate and view previous transactions.")

    card_number = st.text_input("Credit Card Number", placeholder="Enter credit card number (e.g., 4532015112830366)")
    
    if st.button("Validate and Proceed"):
        if not card_number:
            st.error("Please enter a credit card number.")
            return
        
        # Check if card is blocked
        if card_number in BLOCKED_CARDS:
            st.error("❌ This credit card is blocked.")
            return
        
        # Validate using Luhn algorithm
        if not luhn_check(card_number):
            st.error("❌ Invalid credit card number.")
            return
        
        # Fetch and display previous transactions
        transactions = fetch_transactions(card_number)
        if not transactions.empty:
            st.success("✅ Valid credit card. Previous transactions found:")
            st.dataframe(transactions)
        else:
            st.warning("✅ Valid credit card. No previous transactions found.")

        # Store validated card number
        st.session_state.credit_card_number = card_number
        st.success("Credit card validated successfully!")

# --- Main Application ---
def main_app():
    st.title("Credit Card Fraud Fraud Prediction")
    st.sidebar.header("Navigation")
    # Updated navigation to include Credit Card Validation as the last option
    page = st.sidebar.radio("Go to", ["Home", "Upload Data", "Train Model", "Predict", "Visualizations", "Credit Card Validation"])

    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'scaler' not in st.session_state:
        st.session_state.scaler = None
    if 'credit_card_number' not in st.session_state:
        st.session_state.credit_card_number = ""

    if page == "Home":
        st.write(f"""
        Welcome to the Credit Card Fraud Transaction Prediction {' for card number ' + st.session_state.credit_card_number if st.session_state.credit_card_number else ''}.
        This application allows you to:
        """)

    elif page == "Upload Data":
        st.header("Upload Transaction Data")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file:
            st.session_state.df = load_data(uploaded_file)
            if st.session_state.df is not None:
                st.session_state.df, st.session_state.scaler = preprocess_data(st.session_state.df)
                st.write("Data Preview:")
                st.dataframe(st.session_state.df.head())
            else:
                st.error("Failed to load data.")

    elif page == "Train Model":
        st.header("Train Machine Learning Model")
        if st.session_state.df is None:
            st.warning("Please upload data first.")
        else:
            if 'Class' in st.session_state.df.columns:
                X = prepare_features(st.session_state.df)
                y = st.session_state.df['Class']
                if st.button("Train Model"):
                    try:
                        st.session_state.model, X_test, y_test = train_model(X, y)
                        accuracy = evaluate_model(st.session_state.model, X_test, y_test)
                        st.success(f"Model trained successfully! Accuracy: {accuracy:.2f}")
                    except Exception as e:
                        st.error(f"Error training model: {e}")
            else:
                st.error("Dataset is missing the 'Class' column.")

    elif page == "Predict":
        st.header("Transaction Prediction")
        if st.session_state.df is None:
            st.warning("Please upload data first.")
        elif st.session_state.model is None:
            st.warning("Please train a model first.")
        else:
            st.subheader("Select Transaction by Index")
            max_index = len(st.session_state.df) - 1
            index = st.number_input(f"Enter transaction index (0 to {max_index})", min_value=0, max_value=max_index, value=0, step=1)
            
            if st.button("Predict"):
                try:
                    # Extract transaction at the given index
                    input_data = st.session_state.df.iloc[[index]].copy()
                    
                    # Preprocess the data (already preprocessed, but ensure features are correct)
                    input_features = prepare_features(input_data)
                    
                    # Predict
                    prediction = st.session_state.model.predict(input_features)
                    prediction_label = "Fraud" if prediction[0] == 1 else "Non-Fraud"
                    
                    # Check for anomaly
                    anomaly = detect_anomalies(input_features)
                    anomaly_label = "Anomaly" if anomaly[0] == -1 else "Normal"
                    
                    # Display prediction results
                    st.write(f"**Prediction**: {prediction_label} (0 = Non-Fraud, 1 = Fraud)")
                    st.write(f"**Anomaly Status**: {anomaly_label}")
                    
                    # Display all transaction details
                    st.subheader("Transaction Details")
                    details = input_data.to_dict(orient='records')[0]
                    for key, value in details.items():
                        st.write(f"**{key}**: {value}")
                    
                except Exception as e:
                    st.error(f"Error making prediction: {e}")

    elif page == "Visualizations":
        st.header("Data Visualizations")
        if st.session_state.df is None:
            st.warning("Please upload data first.")
        else:
            try:
                plot_transaction_distribution(st.session_state.df)
                X = prepare_features(st.session_state.df)
                anomalies = detect_anomalies(X)
                plot_anomaly_scatter(st.session_state.df, anomalies)
                if st.session_state.model is not None:
                    plot_feature_importance(st.session_state.model, X.columns)
            except Exception as e:
                st.error(f"Error generating visualizations: {e}")

    elif page == "Credit Card Validation":
        credit_card_validation_page()

# --- Main Application Flow ---
def main():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        st.title("Login to Credit Card Fraud Prediction")
        st.markdown("""
        Please enter your username and password to access the application.
        """)
        username = st.text_input("Username", placeholder="Enter username")
        password = st.text_input("Password", type="password", placeholder="Enter password")
        if st.button("Login"):
            if check_login(username, password):
                st.session_state.logged_in = True
                st.success("Login successful! Redirecting to the main application...")
                st.rerun()
            else:
                st.error("Invalid username or password. Please try again.")
    else:
        main_app()

if __name__ == "__main__":
    main()