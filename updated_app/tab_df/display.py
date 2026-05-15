import streamlit as st
import pandas as pd


def overview(df):
    """Displays overview information of the DataFrame in a formatted table."""

    # Create a summary DataFrame
    summary_data = {
        "Description": [
            "Number of Rows",
            "Number of Columns",
            "Number of Duplicated Rows",
            "Number of Rows with Missing Values"
        ],
        "Value": [
            df.shape[0],
            df.shape[1],
            df.duplicated().sum(),
            df.isnull().any(axis=1).sum()
        ]
    }
    summary_df = pd.DataFrame(summary_data)

    # Display the summary table
    st.write("### Dataframe")
    st.table(summary_df)

    st.write("### Column Information")

    # Display column details with data type and memory usage
    try:
        col_info = pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes,
            "Memory Usage (KB)": df.memory_usage(deep=True, index=False) / 1024  # Exclude index memory
        }).reset_index(drop=True)
    except ValueError as e:
        st.error(f"Error generating column information: {e}")
        col_info = pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes,
            "Memory Usage (KB)": ["N/A" for _ in df.columns]  # Use "N/A" as a fallback
        })

    st.table(col_info)

    st.write("### Explore Dataset")

    # Interactive slider and radio button
    row_count = st.slider("Select number of rows to display", min_value=5, max_value=50, value=10)
    display_logic = st.radio("Display logic", options=["Head", "Tail", "Sample"], horizontal=True)

    if display_logic == "Head":
        st.write(df.head(row_count))
    elif display_logic == "Tail":
        st.write(df.tail(row_count))
    else:
        st.write(df.sample(row_count))
