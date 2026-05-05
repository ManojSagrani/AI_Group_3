import streamlit as st
import pandas as pd
from tab_text import logics


def text_serie(df):
    """Displays information about columns with text data in the DataFrame."""

    # Get text columns
    text_columns = logics.get_text_columns(df)

    if text_columns.empty:
        st.write("No text columns found in the dataset.")
        return

    # Text Column Selection
    selected_column = st.selectbox("Which text column do you want to explore?", text_columns.columns)

    if selected_column:
        st.write("### Text Column")

        # Get statistics for the selected text column
        text_stats_df = logics.get_text_statistics(df, selected_column)
        st.table(text_stats_df)

        st.write("### Most Frequent Values")
        freq_df = logics.get_top_frequent_values(df, selected_column)
        st.dataframe(freq_df)
