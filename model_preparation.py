# scripts/model_preparation.py

import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def prepare_data_for_training(merged_csv: str, processed_csv: str) -> tuple:
    """
    Prepares the dataset for training by selecting features and encoding labels.
    
    Args:
        merged_csv (str): Path to the merged data CSV.
        processed_csv (str): Path to save the processed data CSV.
        
    Returns:
        tuple: Feature matrix X and target vector y.
    """
    try:
        logging.info(f"Loading merged data from {merged_csv}")
        df = pd.read_csv(merged_csv)
        
        # Select relevant features
        feature_columns = ['AI_Adoption_Score', 'GDP (USD Billion)', 'Population (Millions)', 
                           'GDP per Capita (USD)', 'Unemployment Rate (%)', 'Literacy Rate (%)', 
                           'Life Expectancy (Years)', 'Inflation Rate (%)', 'Poverty Rate (%)']
        
        X = df[feature_columns]
        
        # Encode target variable
        le = LabelEncoder()
        y = le.fit_transform(df['Policy_Focus'])
        
        # Save processed data
        processed_df = df[['Policy_Focus'] + feature_columns]
        processed_df.to_csv(processed_csv, index=False)
        logging.info(f"Processed data saved to {processed_csv}")
        
        return X, y
    
    except Exception as e:
        logging.error(f"Error preparing data for training: {e}")
        return None, None

if __name__ == "__main__":
    merged_csv_path = 'data/merged_data.csv'
    processed_csv_path = 'data/processed_data_for_training.csv'
    
    X, y = prepare_data_for_training(merged_csv_path, processed_csv_path)
    print(X.head())
    print(y[:5])
