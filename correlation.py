import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from io import StringIO

# NLP imports
import nltk
from wordcloud import WordCloud

# Ensure NLTK data is available
nltk.download('stopwords')
from nltk.corpus import stopwords

# -------------------------
# Helper functions
# -------------------------
def load_data(uploaded_file):
    """
    Carga el archivo CSV o Excel y devuelve un DataFrame.
    """
    if uploaded_file.name.endswith('.csv'):
        return pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(('.xls', '.xlsx')):
        return pd.read_excel(uploaded_file)
    else:
        st.error("Formato de archivo no soportado.")
        return None


def show_correlation(df):
    """
    Muestra la matriz de correlación de variables numéricas.
    """
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if len(numeric_cols) < 2:
        st.warning("No hay suficientes columnas numéricas para correlación.")
        return

    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots()
    cax = ax.matshow(corr, cmap='coolwarm')
    fig.colorbar(cax)
    ax.set_xticks(range(len(numeric_cols)))
    ax.set_yticks(range(len(numeric_cols)))
    ax.set_xticklabels(numeric_cols, rotation=90)
    ax.set_yticklabels(numeric_cols)
    st.pyplot(fig)


def show_text_analysis(df):
    """
    Genera una nube de palabras de la columna de texto seleccionada.
    """
    text_columns = df.select_dtypes(include=['object']).columns.tolist()
    if not text_columns:
        st.warning("No hay columnas de texto para analizar.")
        return

    col = st.selectbox("Selecciona columna de texto", text_columns)
    text = " ".join(df[col].dropna().astype(str))

    # Preparar stopwords
    stops = set(stopwords.words('spanish'))
    wordcloud = WordCloud(
        width=800, height=400,
        background_color='white',
        stopwords=stops
    ).generate(text)

    st.image(wordcloud.to_array(), use_column_width=True)

# -------------------------
# Main
# -------------------------

def main():
    st.title("Análisis de Encuesta: Correlación & Texto Abierto")
    st.markdown("Sube tu archivo de respuestas para empezar.")

    uploaded_file = st.file_uploader(
        "Selecciona un archivo (.csv, .xls, .xlsx)", type=['csv', 'xls', 'xlsx']
    )

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        if df is None:
            return

        st.subheader("Vista previa de datos")
        st.dataframe(df.head())

        # Sidebar para elegir análisis
        st.sidebar.title("Opciones de Análisis")
        analysis = st.sidebar.radio(
            "Selecciona análisis:",
            ("Correlación", "Texto Abierto")
        )

        if analysis == "Correlación":
            st.subheader("Matriz de Correlación")
            show_correlation(df)
        else:
            st.subheader("Análisis de Texto Abierto")
            show_text_analysis(df)

if __name__ == "__main__":
    main()
