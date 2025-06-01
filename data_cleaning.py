import pandas as pd
import numpy as np
from typing import Dict, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clean_market_data(market_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Clean and preprocess market data for multiple stocks.
    
    Args:
        market_data (Dict[str, pd.DataFrame]): Dictionary of stock dataframes
        
    Returns:
        Dict[str, pd.DataFrame]: Dictionary of cleaned stock dataframes
    """
    cleaned_data = {}
    
    for stock_name, df in market_data.items():
        logger.info(f"Cleaning market data for {stock_name}")
        
        # Make a copy to avoid modifying original data
        df_clean = df.copy()
        
        # 1. Handle missing values
        # Forward fill for price data
        price_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']
        df_clean[price_columns] = df_clean[price_columns].fillna(method='ffill')
        
        # Fill remaining missing values with backward fill
        df_clean[price_columns] = df_clean[price_columns].fillna(method='bfill')
        
        # Fill volume missing values with 0
        df_clean['Volume'] = df_clean['Volume'].fillna(0)
        
        # 2. Calculate returns and volatility
        df_clean['Daily_Return'] = df_clean['Adj Close'].pct_change()
        df_clean['Volatility_20d'] = df_clean['Daily_Return'].rolling(window=20).std() * np.sqrt(252)
        
        # 3. Remove outliers using IQR method
        for col in price_columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Replace outliers with NaN and then forward fill
            df_clean.loc[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound), col] = np.nan
            df_clean[col] = df_clean[col].fillna(method='ffill')
        
        # 4. Ensure data is sorted by date
        df_clean = df_clean.sort_index()
        
        # 5. Remove any remaining NaN values
        df_clean = df_clean.dropna()
        
        cleaned_data[stock_name] = df_clean
        
    return cleaned_data

def clean_customer_data(customer_data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess customer profile data.
    
    Args:
        customer_data (pd.DataFrame): Customer profile dataframe
        
    Returns:
        pd.DataFrame: Cleaned customer profile dataframe
    """
    logger.info("Cleaning customer data")
    
    # Make a copy to avoid modifying original data
    df_clean = customer_data.copy()
    
    # 1. Handle missing values
    # Fill missing age with median
    df_clean['age'] = df_clean['age'].fillna(df_clean['age'].median())
    
    # Fill missing income with median
    df_clean['income'] = df_clean['income'].fillna(df_clean['income'].median())
    
    # Fill missing portfolio value with median
    df_clean['current_portfolio_value'] = df_clean['current_portfolio_value'].fillna(
        df_clean['current_portfolio_value'].median()
    )
    
    # Fill categorical variables with mode
    categorical_columns = ['risk_tolerance', 'investment_horizon', 'investment_goal']
    for col in categorical_columns:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
    
    # 2. Handle outliers
    # Remove unrealistic ages (below 18 or above 100)
    df_clean = df_clean[(df_clean['age'] >= 18) & (df_clean['age'] <= 100)]
    
    # Remove negative income or portfolio values
    df_clean = df_clean[(df_clean['income'] > 0) & (df_clean['current_portfolio_value'] > 0)]
    
    # 3. Standardize categorical variables
    df_clean['risk_tolerance'] = df_clean['risk_tolerance'].str.capitalize()
    df_clean['investment_horizon'] = df_clean['investment_horizon'].str.capitalize()
    df_clean['investment_goal'] = df_clean['investment_goal'].str.capitalize()
    
    # 4. Create additional features
    # Calculate income to portfolio ratio
    df_clean['income_portfolio_ratio'] = df_clean['income'] / df_clean['current_portfolio_value']
    
    # Create age groups
    df_clean['age_group'] = pd.cut(
        df_clean['age'],
        bins=[0, 30, 45, 60, 100],
        labels=['Young', 'Middle', 'Senior', 'Elderly']
    )
    
    return df_clean

def align_data_frequencies(market_data: Dict[str, pd.DataFrame], frequency: str = 'D') -> Dict[str, pd.DataFrame]:
    """
    Align all market data to the same frequency.
    
    Args:
        market_data (Dict[str, pd.DataFrame]): Dictionary of stock dataframes
        frequency (str): Target frequency ('D' for daily, 'M' for monthly, etc.)
        
    Returns:
        Dict[str, pd.DataFrame]: Dictionary of aligned stock dataframes
    """
    logger.info(f"Aligning market data to {frequency} frequency")
    
    aligned_data = {}
    
    for stock_name, df in market_data.items():
        # Resample data to target frequency
        df_aligned = df.resample(frequency).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Adj Close': 'last',
            'Volume': 'sum',
            'Daily_Return': 'sum',
            'Volatility_20d': 'last'
        })
        
        aligned_data[stock_name] = df_aligned
    
    return aligned_data

def main():
    """
    Main function to demonstrate data cleaning process.
    """
    try:
        # Load market data
        stocks = ['Apple', 'Microsoft', 'Google', 'Amazon', 'Netflix']
        market_data = {}
        
        for stock in stocks:
            df = pd.read_csv(f'data/{stock}.csv')
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            market_data[stock] = df
        
        # Load customer data
        customer_data = pd.read_csv('data/customer_profiles.csv')
        
        # Clean market data
        cleaned_market_data = clean_market_data(market_data)
        
        # Clean customer data
        cleaned_customer_data = clean_customer_data(customer_data)
        
        # Align market data frequencies
        aligned_market_data = align_data_frequencies(cleaned_market_data, frequency='D')
        
        # Save cleaned data
        for stock_name, df in aligned_market_data.items():
            df.to_csv(f'data/cleaned_{stock_name}.csv')
        
        cleaned_customer_data.to_csv('data/cleaned_customer_profiles.csv', index=False)
        
        logger.info("Data cleaning completed successfully")
        
    except Exception as e:
        logger.error(f"Error during data cleaning: {str(e)}")
        raise

if __name__ == "__main__":
    main() 