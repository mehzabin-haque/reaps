# scripts/model_trainer.py

import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

def train_and_evaluate(processed_csv: str, model_save_path: str):
    """
    Trains a Random Forest classifier and evaluates its performance.
    
    Args:
        processed_csv (str): Path to the processed data CSV.
        model_save_path (str): Path to save the trained model.
    """
    try:
        logging.info(f"Loading processed data from {processed_csv}")
        df = pd.read_csv(processed_csv)
        
        # Separate features and target
        X = df.drop('Policy_Focus', axis=1)
        y = df['Policy_Focus']
        
        # Encode target labels
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
        
        # Initialize and train the model
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        logging.info("Random Forest model trained.")
        
        # Make predictions
        y_pred = clf.predict(X_test)
        
        # Evaluate the model
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        logging.info(f"Model Accuracy: {acc:.2f}")
        logging.info(f"Classification Report:\n{report}")
        
        # Save the model
        joblib.dump({'model': clf, 'label_encoder': le}, model_save_path)
        logging.info(f"Trained model saved to {model_save_path}")
        
    except Exception as e:
        logging.error(f"Error during model training: {e}")

if __name__ == "__main__":
    processed_csv_path = 'data/processed_data_for_training.csv'
    model_save_path = 'models/random_forest_model.joblib'
    
    train_and_evaluate(processed_csv_path, model_save_path)
