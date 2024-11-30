# scripts/data_processing.py

import os
import pandas as pd
import logging

# Import utility functions from utils.py
from utils import (
    extract_texts_from_directory,
    preprocess_text,
    extract_features,
    load_benchmark_data,
    merge_with_benchmarks,
    scale_features,
    prepare_dataset,
    train_model,
    save_model
)

def main():
    # Setup Logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Define paths
    policies_dir = os.path.join('..', 'data', 'policies')
    benchmark_csv = os.path.join('..', 'data', 'benchmarks', 'benchmark_data.csv')
    socio_economic_csv = os.path.join('..', 'data', 'socio_economic_data.csv')
    processed_data_csv = os.path.join('..', 'data', 'processed_data.csv')
    model_save_path = os.path.join('..', 'models', 'random_forest_model.joblib')
    
    # Step 1: Text Extraction
    logging.info("Starting text extraction from policy documents...")
    policies_texts = extract_texts_from_directory(policies_dir)
    
    # Step 2: Preprocessing
    logging.info("Starting text preprocessing...")
    policies_preprocessed = {k: preprocess_text(v) for k, v in policies_texts.items()}
    
    # Step 3: Feature Extraction
    logging.info("Starting feature extraction...")
    policies_features = {}
    for policy_name, text in policies_preprocessed.items():
        features = extract_features(text)
        policies_features[policy_name] = features
    
    # Convert features to DataFrame
    logging.info("Converting features to DataFrame...")
    data_records = []
    for policy_name, features in policies_features.items():
        record = {
            'Policy_Name': policy_name,
            'Country': extract_country_from_entities(features['entities']),  # Implement this function as needed
            'Year': extract_year_from_policy(policy_name),  # Implement this function based on policy metadata
            'Embeddings': features['embeddings'],
            'Entities': features['entities']
        }
        data_records.append(record)
    
    policies_df = pd.DataFrame(data_records)
    
    # Step 4: Load Benchmark Data
    logging.info("Loading benchmark data...")
    benchmark_df = load_benchmark_data(benchmark_csv)
    
    # Step 5: Merge with Benchmarks
    logging.info("Merging policy features with benchmark data...")
    merged_df = merge_with_benchmarks(policies_df, benchmark_df)
    
    # Step 6: Scale Numerical Features
    numerical_features = ['GDP (USD Billion)', 'Population (Millions)', 'GDP per Capita (USD)', 
                          'Unemployment Rate (%)', 'Literacy Rate (%)', 'Life Expectancy (Years)', 
                          'Inflation Rate (%)', 'Poverty Rate (%)']
    logging.info("Scaling numerical features...")
    merged_df = scale_features(merged_df, numerical_features)
    
    # Step 7: Save Processed Data
    logging.info(f"Saving processed data to {processed_data_csv}...")
    merged_df.to_csv(processed_data_csv, index=False)
    logging.info("Processed data saved successfully.")
    
    # Step 8: Load Socio-Economic Data and Merge
    logging.info("Loading socio-economic data...")
    socio_economic_df = pd.read_csv(socio_economic_csv)
    final_df = pd.merge(merged_df, socio_economic_df, on=['Country', 'Year'], how='left')
    logging.info("Merged with socio-economic data.")
    
    # Step 9: Model Training
    target_column = 'Policy_Focus'  # Ensure this column exists in your data
    logging.info("Preparing dataset for model training...")
    X, y = prepare_dataset(final_df, target_column)
    
    logging.info("Training the model...")
    model = train_model(X, y)
    
    # Step 10: Save the Trained Model
    logging.info(f"Saving the trained model to {model_save_path}...")
    save_model(model, model_save_path)
    logging.info("Model training and saving completed successfully.")

def extract_country_from_entities(entities: list) -> str:
    """
    Extracts the country name from the list of entities.

    Args:
        entities (list): List of tuples containing entity text and label.

    Returns:
        str: Extracted country name.
    """
    for ent_text, ent_label in entities:
        if ent_label == 'GPE':  # GPE: Geopolitical Entity
            return ent_text
    return 'Unknown'

def extract_year_from_policy(policy_name: str) -> int:
    """
    Extracts the year from the policy name or assigns a default.

    Args:
        policy_name (str): Name of the policy document.

    Returns:
        int: Year of the policy.
    """
    # Implement logic to extract year from policy name or metadata
    # For simplicity, assigning 2022 as default
    return 2022

if __name__ == "__main__":
    main()
