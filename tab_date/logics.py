import pandas as pd

def get_datetime_columns(df):
    """Returns columns with datetime data types or columns with text that may represent dates."""
    datetime_columns = df.select_dtypes(include=['datetime64[ns]'])
    if datetime_columns.empty:
        text_columns = df.select_dtypes(include=['object'])
        return text_columns
    return datetime_columns

def convert_to_datetime(df, column):
    """Converts a column to datetime if it is not already in datetime format."""
    if not pd.api.types.is_datetime64_any_dtype(df[column]):
        df[column] = pd.to_datetime(df[column], errors='coerce')
    return df

def get_datetime_statistics(df, column):
    """Generates statistics for a selected datetime column."""
    valid_dates = df[column].dropna()

    # Ensure `today` matches the timezone of `valid_dates`
    if valid_dates.dt.tz:
        today = pd.Timestamp.now(tz=valid_dates.dt.tz)
    else:
        today = pd.Timestamp.now()

    stats = {
        "Number of Unique Values": valid_dates.nunique(),
        "Number of Rows with Missing Values": df[column].isnull().sum(),
        "Number of Weekend Dates": valid_dates.dt.weekday.isin([5, 6]).sum(),
        "Number of Weekday Dates": (~valid_dates.dt.weekday.isin([5, 6])).sum(),
        "Number of Dates in Future": (valid_dates > today).sum(),
        "Number of Rows with 1900-01-01": (valid_dates == "1900-01-01").sum(),
        "Number of Rows with 1970-01-01": (valid_dates == "1970-01-01").sum(),
        "Minimum Value": valid_dates.min(),
        "Maximum Value": valid_dates.max()
    }
    return pd.DataFrame(list(stats.items()), columns=["Description", "Value"])

def get_top_frequent_dates(df, column, top_n=20):
    """Returns the top N most frequent dates with their occurrences and percentages."""
    date_counts = df[column].value_counts().head(top_n)
    freq_df = pd.DataFrame({
        "Value": date_counts.index,
        "Occurrences": date_counts.values,
        "Percentage": (date_counts / len(df) * 100).round(2)
    })
    return freq_df
