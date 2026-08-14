import requests
import pandas as pd
from typing import List, Dict, Any

def fetch_total_counts(start_date: str, end_date: str) -> List[Dict[str, Any]]:
    """
    Fetches the total number of samples per day within a specified date range.

    Args:
        start_date: The start date of the time period (YYYY-MM-DD).
        end_date: The end date of the time period (YYYY-MM-DD).

    Returns:
        A list of dictionaries containing 'date' and 'count' for each day.
        Returns an empty list if the query fails.
    """
    lapis_host = "https://lapis.cov-spectrum.org/open/v2"
    endpoint = "/sample/aggregated"
    url = lapis_host + endpoint
    
    query = {
        "dateFrom": start_date,
        "dateTo": end_date,
        "fields": ["date"]
    }
    
    try:
        http_response = requests.post(url, json=query)
        http_response.raise_for_status()
        json_dict = http_response.json()
        
        return json_dict.get("data", [])

    except requests.exceptions.RequestException as e:
        print(f"API query for total counts failed: {e}")
        return []

# --- Main Script Logic ---
global_min_date = "2020-04-27"
global_max_date = "2023-03-13"

# Fetch the data from the API
total_counts_data = fetch_total_counts(global_min_date, global_max_date)

if total_counts_data:
    # Create the pandas DataFrame and save to CSV
    df_total = pd.DataFrame(total_counts_data)
    df_total.columns = ['total_count', 'collection_date'] # Rename columns for clarity
    
    # Define the output path for the single CSV file
    output_path = ""/processed_data/S10_covspectrum_analysis/global_covspectrum_sample_counts.csv"
    
    df_total.to_csv(output_path, index=False)
else:
    print("Failed to fetch total sample counts.")
