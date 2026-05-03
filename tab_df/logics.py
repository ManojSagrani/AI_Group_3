# tab_df/logics.py
import pandas as pd

def load_csv(uploaded_file):
    """Load CSV file into a DataFrame."""
    return pd.read_csv(uploaded_file)
