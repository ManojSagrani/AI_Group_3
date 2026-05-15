import streamlit as st
import pandas as pd
import altair as alt
from tab_numeric import logics


def numeric_serie(df):
    """Displays information and analysis for numeric columns in the DataFrame."""

    # Get numeric columns
    numeric_columns = logics.get_numeric_columns(df)

    if numeric_columns.empty:
        st.write("No numeric columns found in the dataset.")
        return

    # Numeric Column Selection
    selected_column = st.selectbox("Which numeric column do you want to explore?", numeric_columns.columns)

    if selected_column:
        st.write("### Numeric Column")

        # Display statistics for the selected column
        stats_df = logics.get_numeric_statistics(df, selected_column)
        st.table(stats_df)

        # Display histogram using Altair
        st.write("### Histogram")
        histogram = alt.Chart(df).mark_bar().encode(
            alt.X(selected_column, bin=alt.Bin(maxbins=30), title=f"{selected_column} (binned)"),
            alt.Y('count()', title="Count of Records")
        ).properties(width=600, height=400)
        st.altair_chart(histogram, use_container_width=True)

        # Display most frequent values
        st.write("### Most Frequent Values")
        freq_df = logics.get_top_frequent_values(df, selected_column)
        st.dataframe(freq_df)
