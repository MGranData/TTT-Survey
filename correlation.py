import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import plotly.express as px
import nltk
from wordcloud import WordCloud

# Descargar stopwords de NLTK
nltk.download('stopwords')
from nltk.corpus import stopwords

# -------------------------
# Funciones auxiliares
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
    Muestra la matriz de correlación usando Plotly para mejor legibilidad.
    """
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if len(numeric_cols) < 2:
        st.warning("Not enough numeric columns to calculate correlation.")
        return

    corr = df[numeric_cols].corr()
    # Etiquetas acortadas para display y full label en hover
    labels_short = [col if len(col) < 20 else col[:17] + '…' for col in numeric_cols]

    fig = px.imshow(
        corr,
        x=labels_short,
        y=labels_short,
        text_auto=True,
        aspect='equal',
        title="Correlation matrix"
    )
    fig.update_traces(hovertemplate='Variable: %{x}<br>Correlación: %{z:.2f}')
    st.plotly_chart(fig, use_container_width=True)


def show_text_analysis(df):
    """
    Genera una nube de palabras de la columna de texto seleccionada.
    """
    text_columns = df.select_dtypes(include=['object']).columns.tolist()
    if not text_columns:
        st.warning("There are no text columns to analyze.")
        return

    col = st.selectbox("Select a text column", text_columns)
    text = " ".join(df[col].dropna().astype(str))

    # Preparar stopwords en español e inglés
    stops_es = set(stopwords.words('spanish'))
    stops_en = set(stopwords.words('english'))
    stops = stops_es.union(stops_en)
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        stopwords=stops
    ).generate(text)

    # Mostrar la nube de palabras usando use_container_width
    st.image(wordcloud.to_array(), use_container_width=True)

# -------------------------
# Main
# -------------------------

def main():
    st.title("Survey Analysis: Correlation & Open Text")
    st.markdown("Upload your answer file to begin.")

    uploaded_file = st.file_uploader(
        "Select a file (.csv, .xls, .xlsx)",
        type=['csv', 'xls', 'xlsx']
    )

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        if df is None:
            return

        st.subheader("Data preview")
        st.dataframe(df.head())

        st.sidebar.title("Analysis options")
        analysis = st.sidebar.radio(
            "Select analysisis:",
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
