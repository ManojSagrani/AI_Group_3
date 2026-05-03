import pandas as pd

def get_text_columns(df):
    """Returns columns with text data (string/object dtype)."""
    return df.select_dtypes(include=['object'])

def get_text_statistics(df, column):
    """Generates statistics for a selected text column."""
    stats = {
        "Number of Unique Values": df[column].nunique(),
        "Number of Missing Values": df[column].isnull().sum(),
        "Most Frequent Value": df[column].mode().values[0] if not df[column].mode().empty else "N/A",
        "Average Length": df[column].dropna().apply(len).mean()
    }
    return pd.DataFrame(list(stats.items()), columns=["Description", "Value"])

def get_top_frequent_values(df, column, top_n=20):
    """Returns the top N most frequent values with their occurrences and percentages."""
    value_counts = df[column].value_counts().head(top_n)
    freq_df = pd.DataFrame({
        "Value": value_counts.index,
        "Occurrences": value_counts.values,
        "Percentage": (value_counts / len(df) * 100).round(2)
    })
    return freq_df
