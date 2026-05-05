import streamlit as st
import pandas as pd
import altair as alt
from tab_date import logics


def datetime_serie(df):
    """Displays information and analysis for datetime columns in the DataFrame."""

    # Get datetime or potential datetime columns
    datetime_columns = logics.get_datetime_columns(df)

    if datetime_columns.empty:
        st.write("No datetime or potential date columns found in the dataset.")
        return  # Exit the function if no datetime columns are found

    # Datetime Column Selection
    selected_column = st.selectbox("Which datetime column do you want to explore?", datetime_columns.columns)

    if selected_column:
        # Convert selected column to datetime if needed
        df = logics.convert_to_datetime(df, selected_column)

        st.write("### Date Column")

        # Display statistics for the selected datetime column
        stats_df = logics.get_datetime_statistics(df, selected_column)
        st.table(stats_df)

        # Display histogram using Altair
        st.write("### Bar Chart")
        histogram = alt.Chart(df).mark_bar().encode(
            alt.X(selected_column + ":T", title=selected_column),
            alt.Y('count()', title="Count of Records")
        ).properties(width=600, height=400)
        st.altair_chart(histogram, use_container_width=True)

        # Display most frequent values
        st.write("### Most Frequent Values")
        freq_df = logics.get_top_frequent_dates(df, selected_column)
        st.dataframe(freq_df)
