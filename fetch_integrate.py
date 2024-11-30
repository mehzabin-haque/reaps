import pandas as pd
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_socio_economic_data(csv_path: str) -> pd.DataFrame:
    """
    Loads the socio-economic data from a CSV file.
    
    Args:
        csv_path (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    try:
        df = pd.read_csv(csv_path)
        logging.info(f"Socio-economic data loaded from {csv_path}")
        return df
    except FileNotFoundError:
        logging.error(f"File not found: {csv_path}")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"An error occurred while loading data: {e}")
        return pd.DataFrame()

def analyze_socio_economic_data(df: pd.DataFrame):
    """
    Performs basic analysis on the socio-economic data.
    
    Args:
        df (pd.DataFrame): DataFrame containing socio-economic data.
    """
    if df.empty:
        logging.warning("No data available for analysis.")
        return
    
    # Example Analysis: Average GDP per Capita
    avg_gdp_per_capita = df['GDP per Capita (USD)'].mean()
    logging.info(f"Average GDP per Capita (USD) across selected countries: {avg_gdp_per_capita:.2f}")
    
    # Example Analysis: Highest and Lowest Unemployment Rates
    highest_unemployment = df.loc[df['Unemployment Rate (%)'].idxmax()]
    lowest_unemployment = df.loc[df['Unemployment Rate (%)'].idxmin()]
    
    logging.info(f"Highest Unemployment Rate: {highest_unemployment['Country']} at {highest_unemployment['Unemployment Rate (%)']}%")
    logging.info(f"Lowest Unemployment Rate: {lowest_unemployment['Country']} at {lowest_unemployment['Unemployment Rate (%)']}%")
    
    # Add more analyses as required

if __name__ == "__main__":
    csv_path = 'data/socio_economic_data.csv'
    socio_economic_df = load_socio_economic_data(csv_path)
    analyze_socio_economic_data(socio_economic_df)
