from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import streamlit as st
import time
import os

modelo_filename = "svm_digits_model.pkl"

st.title("🧠 Entrenamiento de Modelo SVM")
mensaje_espera = st.empty()

# Verificar si el modelo ya existe
if os.path.exists(modelo_filename):
    st.success("✅ El modelo ya ha sido entrenado y guardado como 'svm_digits_model.pkl'.")
    st.balloons()
else:
    mensaje_espera.warning("⏳ Esperando la creación del modelo...")
    with st.spinner("⚙️ Entrenando el modelo SVM, por favor espera..."):
        time.sleep(2)  

        digits = load_digits()
        X = digits.data  # Datos: (n_samples, 64)
        y = digits.target  # Etiquetas: (n_samples,)

        # 2. Escalar los datos
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 3. Dividir en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # 4. Crear y entrenar el modelo SVM
        clf = SVC(kernel="linear")
        clf.fit(X_train, y_train)

        # Guardar el modelo y el scaler juntos en un diccionario
        modelo = {"scaler": scaler, "clf": clf}

        # Serializar con pickle
        with open(modelo_filename, "wb") as f:
            pickle.dump(modelo, f)
    mensaje_espera.empty() 
    st.success("🎉 ¡Modelo creado exitosamente y guardado como 'svm_digits_model.pkl'! 🚀")
    st.balloons()
