import streamlit as st

st.set_page_config(page_title="Actividad: Clasificador de Dígitos con SVM", layout="wide")

# Estilos personalizados
st.markdown(
    """
    <style>
    .big-header {
        font-size: 36px;
        font-weight: bold;
        color: #4B0082;
        text-align: left;
        margin-top: 20px;
    }
    .section-title {
        font-size: 24px;
        color: #006400;
        margin-top: 30px;
        margin-bottom: 10px;
    }
    .code-box {
        background-color: #f0f0f0;
        padding: 10px;
        border-left: 4px solid #4B0082;
        font-family: monospace;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<div class='big-header'>Actividad: Clasificador de Dígitos Manuscritos con Streamlit y SVM</div>", unsafe_allow_html=True)

st.write("""
En esta actividad vas a desarrollar una aplicación web con **Streamlit** que permitirá a un usuario:
- Dibujar un número manuscrito.
- Subir una imagen con un número.
- Obtener la predicción de un modelo entrenado mediante **SVM** (Support Vector Machine).

El objetivo es aplicar procesamiento de imágenes y aprendizaje automático en una aplicación interactiva real.
""")

st.markdown("<div class='section-title'>Archivos proporcionados</div>", unsafe_allow_html=True)
st.markdown("""
- `svm_digits_model.pkl`: contiene el modelo SVM ya entrenado y el `StandardScaler` usado.
- Base del código de la app (opcional o parcialmente entregada).
""")

st.markdown("<div class='section-title'>Tareas a realizar</div>", unsafe_allow_html=True)

st.markdown("#### 1. Crear la interfaz con Streamlit")
st.write("""
Debes mostrar:
- Un **encabezado explicativo** con estilo.
- Un **lienzo (canvas)** para dibujar un número.
- Un botón para **predecir** el número dibujado.
- Un **cargador de imágenes** para hacer predicciones con archivos JPG o PNG.
""")

st.markdown("#### 2. Preprocesar las imágenes")
st.write("""
Recuerda que un modelo SVM no puede trabajar directamente con imágenes. Hay que:
- Convertir la imagen a **escala de grises**.
- Redimensionarla a **8x8 píxeles**, como en el dataset `digits`.
- Escalar los valores de color de **0–255 a 0–16**.
- Aplicar el `StandardScaler` (que se incluye dentro del archivo `.pkl`) para normalizar la imagen.
""")

with st.expander("¿Por qué 8x8 y valores de 0 a 16?"):
    st.write("""
    El modelo fue entrenado con el conjunto `digits` de Scikit-learn, que contiene imágenes de dígitos:
    - En blanco y negro.
    - De tamaño 8x8.
    - Con valores de píxeles entre 0 y 16.
    Si no se ajustan las nuevas imágenes a ese formato, el modelo no sabrá cómo interpretarlas.
    """)

st.markdown("#### 3. Realizar la predicción")
st.write("""
Una vez preprocesada la imagen, debes usar el modelo para predecir el número que contiene.

El archivo `.pkl` incluye:
- El modelo `clf` (SVM).
- El `scaler` (para transformar las imágenes).
""")

st.markdown("<div class='section-title'>Requisitos técnicos</div>", unsafe_allow_html=True)
st.markdown("""
Instala las librerías necesarias con:
""")
st.code("pip install streamlit numpy pillow opencv-python matplotlib scikit-learn streamlit-drawable-canvas", language="bash")

st.markdown("<div class='section-title'> Recomendaciones</div>", unsafe_allow_html=True)
st.write("""
- Usa funciones para organizar el código.
- Asegúrate de controlar errores si no hay imagen o si está vacía.
- Muestra el resultado de forma clara.
- Comenta el código explicando **por qué** haces cada paso.

Evita copiar sin entender. El objetivo es que comprendas cómo se transforma una imagen en un vector, y cómo el modelo lo usa para decidir qué número es.
""")

st.markdown("<div class='section-title'>Criterios de evaluación</div>", unsafe_allow_html=True)
st.markdown("""
| Criterio                                      | Puntos |
|----------------------------------------------|--------|
| Interfaz funcional y clara                    | 2.0    |
| Preprocesamiento correcto de imágenes         | 2.5    |
| Uso correcto del modelo SVM                   | 2.5    |
| Manejo de errores y funcionalidades extra     | 1.5    |
| Claridad del código y buenas prácticas        | 1.5    |
| **Total**                                     | **10** |
""")


