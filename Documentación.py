import streamlit as st


# Esto debe ir primero
st.set_page_config(
    page_title="Inicio - Predicción de Dígitos",
    initial_sidebar_state="collapsed",
    layout="wide"
)

st.title("Modelo SVM (Support Vector Machine)")


st.markdown("""
Las **Máquinas de Vectores de Soporte (SVM)** son modelos de aprendizaje supervisado utilizados principalmente para clasificación, aunque también pueden aplicarse a regresión.

---
""")

st.subheader("🧠 Idea Principal")
st.markdown("""
El objetivo de SVM es **encontrar el mejor límite (frontera)** que separe dos clases, maximizando el espacio (margen) entre ellas.
""")

st.image("https://upload.wikimedia.org/wikipedia/commons/7/72/SVM_margin.png", 
caption="El hiperplano separa las dos clases con el mayor margen posible.", width=400)

st.subheader("🔍 Componentes Clave")
st.markdown("""
- **Hiperplano**: Es la frontera de decisión que separa las clases.
- **Vectores de soporte**: Son los puntos más cercanos al hiperplano. Solo estos puntos afectan directamente a la posición del hiperplano.
- **Margen**: Es la distancia entre el hiperplano y los vectores de soporte. SVM maximiza este margen.

""")

st.subheader("⚙️ Ventajas del modelo SVM")
st.markdown("""
- ✅ Funciona bien en espacios de alta dimensión.
- ✅ Eficaz cuando hay una clara separación entre clases.
- ✅ Usa pocos puntos de datos (vectores de soporte) → eficiente.
""")

st.subheader("🧠 ¿Cómo se aplica SVM a imágenes?")
st.markdown("""
En problemas como la clasificación de dígitos (por ejemplo, el dataset de dígitos de Scikit-learn):

- Cada imagen de 8x8 píxeles se convierte en un vector de 64 características.
- SVM trata de encontrar un hiperplano en ese espacio que separe dígitos diferentes (por ejemplo, 3s de 5s).
- Para múltiples clases (0 a 9), SVM entrena varios clasificadores binarios (uno contra uno o uno contra todos).



Vector de 64 características:  
[[ 0.  0.  5. 13.  9.  1.  0.  0.]
[ 0.  0. 13. 15. 10. 15.  5.  0.]
[ 0.  3. 15.  2.  0. 11.  8.  0.]

[ 0.  4. 12.  0.  0.  8.  8.  0.]
[ 0.  5.  8.  0.  0.  9.  8.  0.]
[ 0.  4. 11.  0.  1. 12.  7.  0.]

[ 0.  2. 14.  5. 10. 12.  0.  0.]
[ 0.  0.  6. 13. 10.  0.  0.  0.]]

Cada número representa la intensidad del píxel, pero el rango está entre 0 y 16.

""")

st.subheader("🎯 ¿Cómo clasifica SVM varios dígitos?")

st.markdown("""
SVM, por defecto, **solo sabe distinguir entre dos clases** (por ejemplo, "¿es un 3 o no lo es?").

Pero cuando queremos clasificar **dígitos del 0 al 9**, tenemos **10 clases distintas**.


Para ello, entrena varios clasificadores.  
SVM entrena **varios modelos pequeños** que comparan solo **dos números a la vez**. Esto se llama:

---

###### 1️⃣ Uno contra todos (OvR)
- Crea un modelo para cada número.
- Cada modelo aprende: **"¿Es un 3 o no lo es?"**, **"¿Es un 5 o no lo es?"**, etc.
- Se hacen 10 modelos (uno por cada dígito).
- El modelo que esté más seguro es el que da la predicción final.

###### 2️⃣ Uno contra uno (OvO)
- Crea un modelo por **cada par posible de números**:  
  "¿Es un 2 o un 3?", "¿Es un 7 o un 9?", etc.
- En total se crean **45 modelos** para los 10 dígitos.
- Todos los modelos votan, y el número con más votos gana.


###### ⚙️ Scikit-learn (la librería que estamos usando) **usa por defecto la estrategia "uno contra uno"**.


""")

st.header("📝 Ejemplo práctico")
st.code("""
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Cargar dataset
digits = load_digits()
X = digits.data
y = digits.target

# Escalado
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# División de datos
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

# Entrenamiento del modelo SVM
clf = SVC(kernel="linear")
clf.fit(X_train, y_train)
""", language="python")

st.subheader("📊 Tipos de kernel")
st.markdown("""
SVM puede usar diferentes funciones para separar los datos:

- `linear`: Línea o plano recto.
- `poly`: Polinómico (curvas).
- `rbf`: Radial Basis Function (muy común para separaciones complejas).
- `sigmoid`: Similar a una red neuronal.

""")



