import streamlit as st
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import pickle
import matplotlib.pyplot as plt
import time

# Configuración de la página
st.set_page_config(
    page_title="Reconocimiento de Dígitos",
    page_icon="✏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para mejorar la apariencia
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    .prediction-result {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        background-color: #E3F2FD;
    }
    .info-box {
        background-color: #F5F5F5;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .stButton button {
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton button:hover {
        background-color: #1565C0;
    }
    .canvas-container {
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    .divider {
        margin: 2rem 0;
        border-bottom: 1px solid #E0E0E0;
    }
</style>
""", unsafe_allow_html=True)

# Función para cargar el modelo
@st.cache_resource
def cargar_modelo():
    try:
        with open("svm_digits_model.pkl", "rb") as f:
            modelo = pickle.load(f)
        return modelo
    except FileNotFoundError:
        st.error("Archivo del modelo no encontrado. Asegúrate de que 'svm_digits_model.pkl' existe en el directorio actual.")
        return None

# Función de preprocesamiento de imagen
def preprocess_image(image):
    # Convertir a escala de grises
    img = Image.fromarray(image.astype('uint8')).convert('L')
    
    # Redimensionar a 8x8 (SVM fue entrenado con imágenes 8x8)
    img = img.resize((8, 8))
    
    # Escalar valores de píxeles para que coincidan con los datos de entrenamiento (rango 0-16)
    img = 16 * (np.array(img) / 255)
    
    # Aplanar la imagen a un vector de 64 elementos
    image = img.flatten().reshape(1, -1)

    # Aplicar el mismo escalador utilizado durante el entrenamiento
    image = scaler.transform(image)

    return image

# Procesar datos del canvas para SVM
def procesar_datos_canvas(image_data):
    if image_data is None:
        return None
    
    # Extraer los datos de la imagen (ignorando el canal alfa)
    image = np.array(image_data[:, :, :3], dtype=np.uint8)
    
    # Procesar la imagen
    imagen_procesada = preprocess_image(image)
    return imagen_procesada

# Función de predicción con confianza
def predecir_con_confianza(image):
    # Obtener predicción
    prediccion = clf.predict(image)[0]
    
    # Obtener valores de la función de decisión para la confianza
    valores_decision = clf.decision_function(image)
    
    # Encontrar la puntuación de confianza para la clase predicha
    # Para SVM multiclase, necesitamos encontrar la confianza para la clase predicha
    puntuaciones_confianza = {}
    for i in range(10):  # Para dígitos 0-9
        # Para cada dígito, obtener su puntuación de confianza
        puntuaciones_confianza[i] = valores_decision[0][i] if len(valores_decision.shape) > 1 else 0
    
    # Normalizar puntuaciones de confianza a 0-100%
    max_puntuacion = max(puntuaciones_confianza.values())
    min_puntuacion = min(puntuaciones_confianza.values())
    rango_puntuacion = max_puntuacion - min_puntuacion
    
    puntuaciones_normalizadas = {}
    for digito, puntuacion in puntuaciones_confianza.items():
        if rango_puntuacion > 0:
            puntuaciones_normalizadas[digito] = ((puntuacion - min_puntuacion) / rango_puntuacion) * 100
        else:
            puntuaciones_normalizadas[digito] = 50  # Valor predeterminado si todas las puntuaciones son iguales
    
    return prediccion, puntuaciones_normalizadas

# Mostrar puntuaciones de confianza como un gráfico de barras
def mostrar_grafico_confianza(puntuaciones_confianza):
    fig, ax = plt.subplots(figsize=(10, 4))
    
    digitos = list(puntuaciones_confianza.keys())
    puntuaciones = list(puntuaciones_confianza.values())
    
    barras = ax.bar(digitos, puntuaciones, color='skyblue')
    
    # Resaltar el dígito predicho
    indice_predicho = puntuaciones.index(max(puntuaciones))
    barras[indice_predicho].set_color('#1E88E5')
    
    ax.set_xlabel('Dígito')
    ax.set_ylabel('Confianza (%)')
    ax.set_title('Confianza de Predicción por Dígito')
    ax.set_xticks(digitos)
    ax.set_ylim(0, 100)
    
    # Añadir etiquetas de valor encima de las barras
    for barra in barras:
        altura = barra.get_height()
        ax.annotate(f'{altura:.1f}%',
                    xy=(barra.get_x() + barra.get_width() / 2, altura),
                    xytext=(0, 3),  # 3 puntos de desplazamiento vertical
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=8)
    
    st.pyplot(fig)

# Aplicación principal
def main():
    # Encabezado
    st.markdown("<h1 class='main-header'>✏️ Reconocimiento de Dígitos Manuscritos</h1>", unsafe_allow_html=True)
    
    # Barra lateral con información
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png", 
                 caption="Ejemplos de Dígitos MNIST")
        st.markdown("### Acerca de esta Aplicación")
        st.markdown("""
        Esta aplicación utiliza un modelo de Máquina de Vectores de Soporte (SVM) entrenado con el conjunto de datos MNIST para reconocer dígitos manuscritos.
        
        Puedes:
        - Dibujar un dígito en el lienzo
        - Subir una imagen de un dígito manuscrito
        
        El modelo predecirá qué dígito (0-9) está en la imagen.
        """)
        
        st.markdown("### Tecnologías Utilizadas")
        st.markdown("""
        - Streamlit para la interfaz web
        - scikit-learn para el modelo SVM
        - PIL y OpenCV para el procesamiento de imágenes
        - Matplotlib para visualización
        """)

    # Cargar el modelo
    modelo = cargar_modelo()
    if modelo is None:
        st.stop()
    
    global scaler, clf
    scaler = modelo["scaler"]
    clf = modelo["clf"]

    # Crear dos columnas para los dos métodos de entrada
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h2 class='sub-header'>Dibuja un Dígito</h2>", unsafe_allow_html=True)
        st.markdown("<div class='info-box'>Usa tu ratón o pantalla táctil para dibujar un dígito (0-9) en el lienzo de abajo.</div>", 
                   unsafe_allow_html=True)
        
        # Lienzo para dibujar
        canvas_result = st_canvas(
            fill_color="black",
            stroke_width=20,
            stroke_color="white",
            background_color="black",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas",
        )
        
        # Botón de predicción para el lienzo
        if st.button("Predecir Dígito Dibujado", key="predecir_canvas"):
            if canvas_result.image_data is not None and not np.all(canvas_result.image_data[:, :, :3] == 0):
                with st.spinner("Procesando..."):
                    # Añadir un pequeño retraso para mejor experiencia de usuario
                    time.sleep(0.5)
                    
                    # Procesar los datos del lienzo
                    img_procesada = procesar_datos_canvas(canvas_result.image_data)
                    
                    # Hacer predicción
                    prediccion, puntuaciones_confianza = predecir_con_confianza(img_procesada)
                    
                    # Mostrar el resultado
                    st.markdown(f"<div class='prediction-result'>Predicción: {prediccion}</div>", 
                               unsafe_allow_html=True)
                    
                    # Mostrar puntuaciones de confianza
                    st.markdown("<h3>Puntuaciones de Confianza</h3>", unsafe_allow_html=True)
                    mostrar_grafico_confianza(puntuaciones_confianza)
            else:
                st.warning("Por favor, dibuja un dígito antes de predecir.")
    
    with col2:
        st.markdown("<h2 class='sub-header'>Subir una Imagen</h2>", unsafe_allow_html=True)
        st.markdown("<div class='info-box'>Sube una imagen de un dígito manuscrito. Para mejores resultados, usa una imagen clara con buen contraste.</div>", 
                   unsafe_allow_html=True)
        
        # Cargador de archivos
        archivo_subido = st.file_uploader("Elige un archivo de imagen", type=["jpg", "jpeg", "png"])
        
        if archivo_subido is not None:
            # Mostrar la imagen subida
            imagen = Image.open(archivo_subido)
            st.image(imagen, caption="Imagen Subida", width=280)
            
            # Procesar y predecir
            with st.spinner("Procesando..."):
                # Añadir un pequeño retraso para mejor experiencia de usuario
                time.sleep(0.5)
                
                # Convertir a array numpy y procesar
                array_imagen = np.array(imagen)
                img_procesada = preprocess_image(array_imagen)
                
                # Hacer predicción
                prediccion, puntuaciones_confianza = predecir_con_confianza(img_procesada)
                
                # Mostrar el resultado
                st.markdown(f"<div class='prediction-result'>Predicción: {prediccion}</div>", 
                           unsafe_allow_html=True)
                
                # Mostrar puntuaciones de confianza
                st.markdown("<h3>Puntuaciones de Confianza</h3>", unsafe_allow_html=True)
                mostrar_grafico_confianza(puntuaciones_confianza)
    
    # Divisor
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    
    # Sección de detalles técnicos
    with st.expander("Detalles Técnicos"):
        st.markdown("""
        ### Preprocesamiento de Imágenes
        
        La imagen de entrada pasa por estos pasos de preprocesamiento:
        1. Conversión a escala de grises
        2. Redimensionamiento a 8x8 píxeles (64 características)
        3. Escalado de valores de píxeles para que coincidan con los datos de entrenamiento
        4. Aplanamiento a un array 1D
        5. Escalado de características utilizando el mismo escalador que durante el entrenamiento
        
        """)

if __name__ == "__main__":
    main()