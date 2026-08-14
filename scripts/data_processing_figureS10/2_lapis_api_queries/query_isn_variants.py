import requests
import pandas as pd
import pathlib
from typing import List, Dict, Any

# --- Constants ---
LAPIS_HOST = "https://lapis.cov-spectrum.org/open/v2"
OUTPUT_DIR = pathlib.Path("/processed_data/S10_covspectrum_analysis/")

def fetch_mutation_counts(mutations_and_types: List[Dict[str, str]], start_date: str, end_date: str) -> List[Dict[str, Any]]:
    """
    Fetches daily counts for a list of nucleotide mutations from the LAPIS API.

    Args:
        mutations_and_types: A list of dictionaries with 'mutation' and 'position_type'.
        start_date: The start date of the time period (YYYY-MM-DD).
        end_date: The end date of the time period (YYYY-MM-DD).

    Returns:
        A list of dictionaries, where each dictionary contains data for a specific mutation.
        Returns an empty list if any query fails.
    """
    url = LAPIS_HOST + "/sample/aggregated"
    all_data = []
    
    for item in mutations_and_types:
        mutation = item['mutation']
        position_type = item['position_type']
        
        query = {
            "nucleotideMutations": [mutation],
            "dateFrom": start_date,
            "dateTo": end_date,
            "fields": ["date"]
        }
        
        try:
            http_response = requests.post(url, json=query)
            http_response.raise_for_status()
            json_dict = http_response.json()
            
            for entry in json_dict.get("data", []):
                entry['mutation'] = mutation
                entry['position_type'] = position_type
                all_data.append(entry)

        except requests.exceptions.RequestException as e:
            print(f"Query for mutation {mutation} failed: {e}")
            return []
            
    return all_data

def get_variants_from_file(file_path: pathlib.Path) -> List[str]:
    """Reads a list of mutation variants from a .txt file."""
    variants = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                # Assuming each line is a single variant, e.g., "249G"
                variant = line.strip()
                if variant:
                    variants.append(variant)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    return variants

# The get_surrounding_positions function is no longer needed.

# --- Main Script Logic ---
global_min_date = "2020-04-27"
global_max_date = "2023-03-13"

centre_protocols = ["NORT_ARTIC_3","NORT_ARTIC_4","NORT_ARTIC_4.1","NORW_ARTIC_4.1","OXON_VeSeq","PHEC_ARTIC_3","SANG_ARTIC_3","SANG_ARTIC_4.1"]

for dgp_id in centre_protocols:
    file_path = OUTPUT_DIR / f"dgp/{dgp_id}_artefacts.txt"
    original_variants = get_variants_from_file(file_path)
    
    if not original_variants:
        continue
    
    # Create the list of mutations with their type. We only have 'original' now.
    mutations_with_types = []
    for variant in original_variants:
        mutations_with_types.append({'mutation': variant, 'position_type': 'original'})

    # Fetch data from the API using the global date range
    aggregated_data_list = fetch_mutation_counts(mutations_with_types, global_min_date, global_max_date)

    if aggregated_data_list:
        df = pd.DataFrame(aggregated_data_list)
        output_path = OUTPUT_DIR / str(dgp_id) / f"consensus_isnv_variants_fetch_global_{dgp_id}.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
    else:
        print(f"No data fetched for centre protocol {dgp_id}.")
