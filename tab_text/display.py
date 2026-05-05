import streamlit as st
from tab_text import logics


def text_serie(df):
    text_columns = logics.get_text_columns(df)
    if text_columns.empty:
        st.info("No text columns found in the dataset.")
        return

    selected_column = st.selectbox("Select a text column to explore", text_columns.columns)
    if not selected_column:
        return

    st.write("### Column Statistics")
    stats_df = logics.get_text_statistics(df, selected_column)
    st.table(stats_df)

    st.write("### Most Frequent Values")
    freq_df = logics.get_top_frequent_values(df, selected_column)
    st.dataframe(freq_df, use_container_width=True)
