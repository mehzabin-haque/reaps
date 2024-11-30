import wbdata
import pandas as pd
from datetime import datetime, date
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_socio_economic_data():
    """
    Fetches socio-economic data from the World Bank API and saves it to a CSV file.
    Compatible with wbdata version 1.0.0.
    """
    # Define the indicators you need
    indicators = {
        'NY.GDP.MKTP.CD': 'GDP (USD Billion)',
        'SP.POP.TOTL': 'Population (Millions)',
        'NY.GDP.PCAP.CD': 'GDP per Capita (USD)',
        'SL.UEM.TOTL.ZS': 'Unemployment Rate (%)',
        'SE.ADT.LITR.ZS': 'Literacy Rate (%)',
        'SP.DYN.LE00.IN': 'Life Expectancy (Years)',
        'FP.CPI.TOTL.ZG': 'Inflation Rate (%)',
        'SI.POV.DDAY': 'Poverty Rate (%)'
    }

    # Define the countries (ISO3 codes)
    countries = ['USA', 'DEU', 'IND', 'BRA', 'NGA', 'CAN', 'AUS', 'FRA', 'JPN', 'ZAF']

    # Define the date range using datetime.date objects
    desired_year = 2022

    try:
        # Fetch the data without 'convert_date' and 'data_date'
        logging.info("Fetching socio-economic data from World Bank...")
        df = wbdata.get_dataframe(
            indicators,
            country=countries,
        )

        # Reset index to turn country and date into columns
        df.reset_index(inplace=True)

        # Check if 'date' column exists
        if 'date' in df.columns:
            # Ensure 'date' is in datetime format
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                df['date'] = pd.to_datetime(df['date'])
            
            # Extract year from 'date' column
            df['Year'] = df['date'].dt.year
            
            # Drop the original 'date' column
            df.drop('date', axis=1, inplace=True)
            
            # Filter the DataFrame for the desired year
            df = df[df['Year'] == desired_year]
            logging.info(f"Filtered data for the year {desired_year}.")
        else:
            logging.warning("'date' column not found in the fetched data.")

        # Rename 'country' to 'Country' if it exists
        if 'country' in df.columns:
            df.rename(columns={'country': 'Country'}, inplace=True)
            logging.info("Renamed 'country' column to 'Country'.")
        else:
            logging.warning("'country' column not found in the fetched data.")

        # Reorder columns for better readability
        cols = ['Country', 'Year'] + [indicator for indicator in indicators.values()]
        # Ensure all required columns are present
        existing_cols = [col for col in cols if col in df.columns]
        df = df[existing_cols]

        # Save the processed DataFrame to CSV
        csv_path = 'socio_economic_data.csv'
        df.to_csv(csv_path, index=False)
        logging.info(f"Socio-economic data saved to {csv_path}")

        # Display the first few rows of the DataFrame
        print(df.head())

    except TypeError as te:
        logging.error(f"TypeError encountered: {te}")
        logging.info("This may be due to incorrect parameter usage. Please check the wbdata documentation for version 1.0.0.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    fetch_socio_economic_data()
