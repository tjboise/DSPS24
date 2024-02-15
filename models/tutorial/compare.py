import pandas as pd

# Function to calculate MAPE
def calculate_mape(actual, forecast):
    actual, forecast = pd.Series(actual), pd.Series(forecast)
    return ((actual - forecast).abs() / actual).mean() * 100
# Function to read CSV and compute MAPE
def compute_mape(csv_file1, csv_file2):
    # Read the CSV files
    df1 = pd.read_csv(csv_file1)
    df2 = pd.read_csv(csv_file2)
    
    # Ensure that 'filename' is of string type to avoid any merge issues
    df1['filename'] = df1['filename'].astype(str)
    df2['filename'] = df2['filename'].astype(str)

    # Merge the dataframes on 'filename'
    merged_df = pd.merge(df1, df2, on='filename', suffixes=('_1', '_2'))
    
    # Calculate MAPE
    mape = calculate_mape(merged_df['res_1'], merged_df['res_2'])
    return mape

# Example usage
csv_file1 = input("first file: ")
csv_file2 = input("second file: ")

mape = compute_mape(csv_file1, csv_file2)
print(f"The MAPE is: {mape}%")
