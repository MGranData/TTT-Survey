import streamlit as st
import pandas as pd
import numpy as np
import re
from PIL import Image
import plotly.express as px

# NLP imports
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
    Carga un archivo CSV o Excel y devuelve un DataFrame.
    """
    if uploaded_file.name.endswith('.csv'):
        return pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(('.xls', '.xlsx')):
        return pd.read_excel(uploaded_file)
    else:
        st.error("Formato de archivo no soportado. Solo .csv, .xls, .xlsx")
        return None


def show_correlation_plotly(df):
    """
    Muestra la matriz de correlación usando Plotly para mejor legibilidad
    y la tabla de preguntas con su número.
    """
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if len(numeric_cols) < 2:
        st.warning("No hay suficientes columnas numéricas para calcular correlación.")
        return

    # Obtener matriz de correlación
    corr = df[numeric_cols].corr()
    labels_short = [col if len(col) < 20 else col[:17] + '…' for col in numeric_cols]

    # Plotly heatmap
    fig = px.imshow(
        corr,
        x=labels_short,
        y=labels_short,
        text_auto=True,
        aspect='equal',
        title="Matriz de Correlación"
    )
    fig.update_traces(hovertemplate='Variable: %{x}<br>Correlación: %{z:.2f}')
    st.plotly_chart(fig, use_container_width=True)

    # Tabla de preguntas
    mapping = []
    for col in numeric_cols:
        m = re.match(r"\s*(\d+)\.\s*(.*)", col)
        if m:
            num, text = m.group(1), m.group(2)
        else:
            num, text = '', col
        mapping.append({'N': num, 'Question': text})
    df_q = pd.DataFrame(mapping)
    st.subheader("Tabla de preguntas")
    st.table(df_q)


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

    # Stopwords español + inglés
    stops_es = set(stopwords.words('spanish'))
    stops_en = set(stopwords.words('english'))
    stops = stops_es.union(stops_en)

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        stopwords=stops
    ).generate(text)

    st.image(wordcloud.to_array(), use_container_width=True)

# -------------------------
# Main
# -------------------------

def main():
    st.title("Análisis de Encuesta: Correlación & Texto Abierto")
    st.markdown("Sube tu archivo de respuestas para empezar.")

    uploaded_file = st.file_uploader(
        "Selecciona un archivo (.csv, .xls, .xlsx)",
        type=['csv', 'xls', 'xlsx']
    )

    if uploaded_file:
        df = load_data(uploaded_file)
        if df is None:
            return

        st.subheader("Vista previa de datos")
        st.dataframe(df.head())

        st.sidebar.title("Opciones de Análisis")
        analysis = st.sidebar.radio(
            "Selecciona análisis:",
            ("Correlación", "Texto Abierto")
        )

        if analysis == "Correlación":
            st.subheader("Matriz de Correlación Interactiva")
            show_correlation_plotly(df)
        else:
            st.subheader("Análisis de Texto Abierto")
            show_text_analysis(df)

if __name__ == "__main__":
    main()
