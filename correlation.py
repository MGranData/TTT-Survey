import streamlit as st
import pandas as pd
import numpy as np
import re
from PIL import Image
import plotly.express as px

# NLP imports
import nltk
from wordcloud import WordCloud

# Download NLTK stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

# -------------------------
# Helper functions
# -------------------------

def load_data(uploaded_file):
    """
    Load a CSV or Excel file and return a DataFrame.
    Supports .csv, .xls, .xlsx
    """
    try:
        if uploaded_file.name.endswith('.csv'):
            # adjust decimal/thousands if needed: decimal=',', thousands='.'
            return pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            return pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Only .csv, .xls, .xlsx are supported.")
    except Exception as e:
        st.error(f"Error loading file: {e}")
    return None


def show_correlation_plotly(df):
    """
    Display an interactive correlation matrix using Plotly,
    allowing the user to select which columns to include.
    Also display a table mapping question numbers to full text.
    """
    # Clean column names
    df.columns = df.columns.str.strip()

    # List all columns and detect numeric dtype for defaults
    all_cols = df.columns.tolist()
    default_numeric = df.select_dtypes(include=[np.number]).columns.tolist()

    # Let the user select which columns to correlate
    selected = st.multiselect(
        "Select columns to include in correlation:",
        options=all_cols,
        default=default_numeric,
        help="Pick at least two numeric columns."
    )

    if len(selected) < 2:
        st.warning("Please select at least two columns for correlation.")
        return

    # Convert selected columns to numeric, coerce errors
    temp_df = df[selected].apply(pd.to_numeric, errors='coerce')
    corr = temp_df.corr()

    # Shorten labels for display, full text on hover
    labels_short = [col if len(col) < 20 else col[:17] + '…' for col in selected]

    # Plot interactive heatmap
    fig = px.imshow(
        corr,
        x=labels_short,
        y=labels_short,
        text_auto=True,
        aspect='equal',
        title="Correlation Matrix"
    )
    fig.update_traces(hovertemplate='Question: %{x}<br>Correlation: %{z:.2f}')
    st.plotly_chart(fig, use_container_width=True)

    # Build question mapping table
    mapping = []
    for original_col, short_label in zip(selected, labels_short):
        match = re.match(r"\s*(\d+)\.\s*(.*)", original_col)
        if match:
            num, text = match.group(1), match.group(2)
        else:
            num, text = '', original_col
        mapping.append({'N': num, 'Question': text})
    df_q = pd.DataFrame(mapping)
    st.subheader("Questions Table")
    # display without index
    st.table(df_q)


def show_text_analysis(df):
    """
    Generate a word cloud from a selected text column.
    """
    text_cols = df.select_dtypes(include=['object']).columns.tolist()
    if not text_cols:
        st.warning("No text columns available for analysis.")
        return

    col = st.selectbox("Select text column:", text_cols)
    text = " ".join(df[col].dropna().astype(str))

    # Combine Spanish and English stopwords
    stops_es = set(stopwords.words('spanish'))
    stops_en = set(stopwords.words('english'))
    stops = stops_es.union(stops_en)

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        stopwords=stops
    ).generate(text)

    # Display word cloud
    st.subheader("Word Cloud")
    st.image(wordcloud.to_array(), use_container_width=True)

# -------------------------
# Main
# -------------------------

def main():
    st.title("Survey Analysis: Correlation & Open Text")
    st.markdown("Upload your responses file to get started.")

    uploaded_file = st.file_uploader(
        "Select a file (.csv, .xls, .xlsx)",
        type=['csv', 'xls', 'xlsx']
    )

    if uploaded_file is None:
        return

    df = load_data(uploaded_file)
    if df is None:
        return

    st.subheader("Data Preview")
    st.dataframe(df.head())

    st.sidebar.title("Analysis Options")
    analysis = st.sidebar.radio(
        "Select analysis:",
        ("Correlation", "Open Text")
    )

    if analysis == "Correlation":
        st.subheader("Interactive Correlation Matrix")
        show_correlation_plotly(df)
    else:
        st.subheader("Open Text Analysis")
        show_text_analysis(df)

if __name__ == "__main__":
    main()
