import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import pickle

def preprocess_image(image):
    # # Convertir a escala de grises
    img = Image.fromarray(image.astype('uint8')).convert('L')
    
    # #Redimensionar, ya que SVM fue entrenado con imagenes 8x8
    img = img.resize((8, 8))
    # #Necesitamos convertirla en una matriz de números, y que 
    # # sean hexadecimales, no de 0 a 255, porque digits usa eso, por eso escalamos
    img = 16 * (np.array(img) / 255)
    # # Aplanar la imagen para que sea un vector de 64 elementos
    image = img.flatten().reshape(1, -1)

    # # Aplicar el mismo scaler que usaste al entrenar
    image = scaler.transform(image)

    return image


def preprocesar_canvas_para_svm(image_data):
    if image_data is None:
        return None
    
    image = np.array(image_data, dtype=np.uint8)
    imagen_scaled = preprocess_image(image)
    return imagen_scaled

#Predice mediante la imagen 
def predict(image):
    prediccion = clf.predict(image)
    return prediccion[0]

# Cargar el modelo desde el archivo
with open("svm_digits_model.pkl", "rb") as f:
    modelo = pickle.load(f)

scaler = modelo["scaler"]
clf = modelo["clf"]

# Create a canvas to draw the digit
st.set_page_config(page_title="Reconocimiento de Dígitos", page_icon=":pencil2:", layout="wide")

# Título con estilo
st.title("Reconocimiento de Dígitos Manuscritos :pencil2:")
st.markdown("<h5 style='text-align: center;'>¡Dibuja un número o sube una imagen para predecir!</h5>", unsafe_allow_html=True)
canvas = st_canvas(
    fill_color="black",
    stroke_width=20,
    stroke_color="white",
    background_color="black",
    height=150,
    width=150,
    drawing_mode="freedraw",
    key="canvas",
)

#Predice con la imagen
if st.button("Predict"):
    if np.all(canvas.image_data[:, :, :3] == 0):
        st.write("Por favor, dibuja un número antes de predecir.")
    else:
        img_processed = preprocesar_canvas_para_svm(canvas.image_data)
        if img_processed is not None:
            prediction = clf.predict(img_processed)
            st.subheader("Predicción")
            st.write(f"El modelo predice que el número es: **{prediction[0]}**")

archivo_subido = st.file_uploader("Sube una imagen manuscrita (JPG o PNG)", type=["jpg", "png"])

if archivo_subido is not None:
    # Mostrar imagen con PIL
    image = Image.open(archivo_subido)
    st.image(image, caption='Imagen subida',  width=150)  
    st.write("")
    image = np.array(image)
    img_processed = preprocess_image(image)
    
    # Realizar predicción
    prediction = predict(img_processed)
    # Make a prediction
    st.subheader(f"✅ El modelo predice que el número es: **{prediction}**")

st.write("Esta app usa OpenCV para procesar imágenes y Scikit-learn para predecir dígitos manuscritos.")


