import streamlit as st
import pandas as pd
import sys
import os

# Add the parent directory to the Python path to locate other packages
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# Import modules from tab_df, tab_text, tab_numeric, and tab_date
from tab_df import display as df_display, logics as df_logics
from tab_text import display as text_display
from tab_numeric import display as numeric_display
from tab_date import display as date_display

# Set page configuration and dark theme
st.set_page_config(page_title="CSV Explorer", layout="wide", initial_sidebar_state="expanded")

# Header section
st.title("CSV Explorer")
st.markdown("### - Streamlit application for performing data exploration on a CSV")

# Sidebar for file upload
st.sidebar.header("Choose a CSV file")
uploaded_file = st.sidebar.file_uploader("Drag and drop file here", type=["csv"])

if uploaded_file:
    # Load CSV data
    df = df_logics.load_csv(uploaded_file)
    st.sidebar.success(f"{uploaded_file.name} successfully loaded!")

    # Tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs(["DataFrame", "Numeric Serie", "Text Serie", "Datetime Serie"])

    # DataFrame Tab
    with tab1:
        df_display.overview(df)

    # Numeric Serie Tab
    with tab2:
        numeric_display.numeric_serie(df)

    # Text Serie Tab
    with tab3:
        text_display.text_serie(df)

    # Datetime Serie Tab
    with tab4:
        pass
        date_display.datetime_serie(df)

else:
    st.sidebar.info("Upload a CSV file to explore its content.")
