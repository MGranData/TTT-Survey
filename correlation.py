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
    """
    if uploaded_file.name.endswith('.csv'):
        return pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(('.xls', '.xlsx')):
        return pd.read_excel(uploaded_file)
    else:
        st.error("Unsupported file format. Only .csv, .xls, .xlsx are supported.")
        return None


def show_correlation_plotly(df):
    """
    Display an interactive correlation matrix using Plotly
    and a table mapping question numbers to their full text.
    """
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if len(numeric_cols) < 2:
        st.warning("Not enough numeric columns to compute correlation.")
        return

    # Compute correlation matrix
    corr = df[numeric_cols].corr()
    # Shorten labels for display, full text on hover
    labels_short = [col if len(col) < 20 else col[:17] + '…' for col in numeric_cols]

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
    for col in numeric_cols:
        match = re.match(r"\s*(\d+)\.\s*(.*)", col)
        if match:
            num, text = match.group(1), match.group(2)
        else:
            num, text = '', col
        mapping.append({'N': num, 'Question': text})
    df_q = pd.DataFrame(mapping).reset_index(drop=True)
    st.subheader("Questions Table")
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

    if uploaded_file:
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
