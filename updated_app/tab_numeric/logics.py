import pandas as pd

def get_numeric_columns(df):
    """Returns columns with numeric data types."""
    return df.select_dtypes(include='number',exclude='datetime')

def get_numeric_statistics(df, column):
    """Generates statistics for a selected numeric column."""
    stats = {
        "Number of Unique Values": df[column].nunique(),
        "Number of Rows with Missing Values": df[column].isnull().sum(),
        "Number of Rows with 0": (df[column] == 0).sum(),
        "Number of Rows with Negative Values": (df[column] < 0).sum(),
        "Average Value": df[column].mean(),
        "Standard Deviation Value": df[column].std(),
        "Minimum Value": df[column].min(),
        "Maximum Value": df[column].max(),
        "Median Value": df[column].median()
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
