import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

# Configuración de la página
st.set_page_config(
    page_title="🔤 Reconocimiento de Letras Manuscritas",
    page_icon="🔤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Importar utilidades personalizadas
from utils.data_loader import (
    load_train_data, 
    load_test_data, 
    get_dataset_info,
    label_to_letter,
    create_sample_images
)
from utils.model_builder import (
    load_model,
    predict_letter,
    model_exists,
    get_model_info
)

# Inicializar session_state - Elemento 4: DEFINICIÓN DE ESTADO DE LA SESIÓN
# Justificación: Mantener datos persistentes durante la sesión del usuario
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'current_prediction' not in st.session_state:
    st.session_state.current_prediction = None
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []
if 'sample_images' not in st.session_state:
    st.session_state.sample_images = None

# Título principal
st.title("🔤 Reconocimiento de Letras Manuscritas")
st.markdown("""
### 🎯 Sistema de Machine Learning con Dataset EMNIST

Esta aplicación utiliza una **Red Neuronal Convolucional (CNN)** entrenada con el dataset **EMNIST Letters** 
para reconocer letras manuscritas (A-Z).

**Características:**
- 📊 Visualización exploratoria de datos
- 🤖 Entrenamiento de modelo CNN
- 🎨 Dibuja letras y obtén predicciones en tiempo real
- 💬 Chatbot asistente con análisis
- 📈 Métricas y estadísticas del modelo
""")

st.info("👈 Usa el **menú lateral** para navegar entre las diferentes páginas de la aplicación")

# Sidebar - Elemento 3: WIDGETS
with st.sidebar:
    st.header("📊 Información del Sistema")
    
    # Información del dataset
    dataset_info = get_dataset_info()
    
    if dataset_info['train_file_exists']:
        st.success("✅ Dataset de entrenamiento encontrado")
    else:
        st.error("❌ Dataset de entrenamiento no encontrado")
    
    if dataset_info['test_file_exists']:
        st.success("✅ Dataset de prueba encontrado")
    else:
        st.error("❌ Dataset de prueba no encontrado")
    
    st.divider()
    
    # Información del modelo
    st.subheader("🤖 Estado del Modelo")
    if model_exists():
        st.success("✅ Modelo entrenado disponible")
        model_info = get_model_info()
        if model_info:
            st.metric("Precisión", f"{model_info['val_accuracy']*100:.2f}%")
            st.metric("Épocas", model_info['epochs_trained'])
    else:
        st.warning("⚠️ No hay modelo entrenado")
        st.info("Ve a la página **Entrenamiento** para crear uno")
    
    st.divider()
    
    # Estadísticas de la sesión
    st.subheader("📈 Sesión Actual")
    st.metric("Predicciones realizadas", len(st.session_state.prediction_history))

# Contenido principal
st.header("🏠 Página Principal")

# Sección de inicio rápido
col1, col2 = st.columns(2)

with col1:
    st.subheader("🚀 Inicio Rápido")
    st.markdown("""
    **1. Visualización de Datos**
    - Explora el dataset EMNIST
    - Visualiza distribución de letras
    - Analiza muestras de imágenes
    
    **2. Entrenamiento del Modelo**
    - Configura parámetros de entrenamiento
    - Entrena la red neuronal CNN
    - Visualiza métricas de rendimiento
    
    **3. Dibuja y Reconoce**
    - Dibuja letras a mano alzada
    - Obtén predicciones en tiempo real
    - Explora las probabilidades
    """)

with col2:
    st.subheader("🎨 Demo Rápida")
    
    # Cargar algunas imágenes de muestra
    if st.button("🔄 Cargar Imágenes de Muestra", width='stretch'):
        with st.spinner("Cargando imágenes..."):
            X_sample, y_sample = create_sample_images(num_samples=6)
            if X_sample is not None:
                st.session_state.sample_images = (X_sample, y_sample)
                st.success("✅ Imágenes cargadas!")
                st.rerun()
    
    # Mostrar imágenes de muestra
    if st.session_state.sample_images is not None:
        X_sample, y_sample = st.session_state.sample_images
        
        # Crear grid de imágenes
        cols = st.columns(3)
        for i in range(min(6, len(X_sample))):
            with cols[i % 3]:
                letter = label_to_letter(y_sample[i])
                st.image(X_sample[i], caption=f"Letra: {letter}", width=100)

# Información del dataset
st.divider()
st.header("📚 Sobre el Dataset EMNIST")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Clases",
        dataset_info['num_classes'],
        help="Número de letras (A-Z)"
    )

with col2:
    st.metric(
        "Tamaño de Imagen",
        "28x28 píxeles",
        help="Imágenes en escala de grises"
    )

with col3:
    if dataset_info.get('train_samples') != 'Desconocido':
        st.metric(
            "Muestras de Entrenamiento",
            f"{dataset_info.get('train_samples', 'N/A'):,}",
            help="Total de imágenes para entrenar"
        )

st.markdown("""
### 📖 Acerca del Dataset

**EMNIST Letters** es una extensión del famoso dataset MNIST, diseñado específicamente para el reconocimiento 
de letras manuscritas. Contiene:

- **26 clases**: Una para cada letra del alfabeto (A-Z)
- **Imágenes en escala de grises**: 28x28 píxeles
- **Escritura real**: Letras escritas a mano por diferentes personas
- **Balanceado**: Distribución equitativa de clases

Este dataset es ideal para aprender y demostrar técnicas de Deep Learning y Computer Vision.
""")

# Sección del chatbot - Elemento 2: CHAT BOT
st.divider()
st.header("💬 Asistente Virtual")

# Función para generar respuestas del chatbot
def generate_chat_response(user_input, dataset_info):
    """
    Genera respuestas del chatbot basadas en el input del usuario
    """
    user_input_lower = user_input.lower()
    
    # Respuestas basadas en comandos comunes
    if "dataset" in user_input_lower or "datos" in user_input_lower:
        return f"""📊 **Información del Dataset EMNIST Letters:**

- **Tipo**: Reconocimiento de letras manuscritas
- **Clases**: {dataset_info['num_classes']} (A-Z)
- **Tamaño de imagen**: {dataset_info['image_shape'][0]}x{dataset_info['image_shape'][1]} píxeles
- **Dataset de entrenamiento**: {'✅ Disponible' if dataset_info['train_file_exists'] else '❌ No encontrado'}
- **Dataset de prueba**: {'✅ Disponible' if dataset_info['test_file_exists'] else '❌ No encontrado'}

El dataset EMNIST es perfecto para aprender técnicas de Deep Learning!
"""
    
    elif "modelo" in user_input_lower or "cnn" in user_input_lower or "red neuronal" in user_input_lower:
        if model_exists():
            model_info = get_model_info()
            if model_info:
                return f"""🤖 **Información del Modelo CNN:**

- **Arquitectura**: Red Neuronal Convolucional (CNN)
- **Precisión en validación**: {model_info['val_accuracy']*100:.2f}%
- **Pérdida en validación**: {model_info['val_loss']:.4f}
- **Épocas de entrenamiento**: {model_info['epochs_trained']}

El modelo está listo para hacer predicciones. Ve a la página **"Dibuja y Reconoce"** para probarlo!
"""
        return """🤖 **Sobre el Modelo CNN:**

El modelo utiliza una arquitectura de Red Neuronal Convolucional con:
- 3 capas convolucionales
- Capas de MaxPooling para reducir dimensionalidad
- Dropout para evitar overfitting
- Capa densa final con activación softmax

Actualmente no hay un modelo entrenado. Ve a la página **"Entrenamiento"** para crear uno.
"""
    
    elif "entrenar" in user_input_lower or "training" in user_input_lower:
        return """🎓 **Entrenamiento del Modelo:**

Para entrenar el modelo:

1. Ve a la página **"🤖 Entrenamiento"** desde el menú lateral
2. Configura los hiperparámetros (épocas, tamaño de muestra, etc.)
3. Haz clic en **"Iniciar Entrenamiento"**
4. Observa las métricas en tiempo real

El entrenamiento puede tardar varios minutos dependiendo del tamaño del dataset.
"""
    
    elif "usar" in user_input_lower or "probar" in user_input_lower or "dibujar" in user_input_lower:
        return """🎨 **Usar el Modelo:**

Para probar el reconocimiento de letras:

1. Ve a la página **"🎨 Dibuja y Reconoce"**
2. Dibuja una letra en el canvas
3. El modelo predecirá qué letra dibujaste
4. Verás las probabilidades para cada letra

¡Es muy divertido y educativo!
"""
    
    elif "páginas" in user_input_lower or "navegación" in user_input_lower:
        return """📚 **Páginas Disponibles:**

1. **🏠 Inicio** - Esta página con información general
2. **📊 Visualización de Datos** - Explora el dataset EMNIST
3. **🤖 Entrenamiento** - Entrena el modelo CNN
4. **🎨 Dibuja y Reconoce** - Prueba el modelo dibujando letras

Usa el menú lateral para navegar entre páginas.
"""
    
    elif "ayuda" in user_input_lower or "help" in user_input_lower:
        return """🤖 **Comandos Disponibles:**

- **"dataset"** o **"datos"** - Información sobre el dataset
- **"modelo"** o **"cnn"** - Información del modelo
- **"entrenar"** - Cómo entrenar el modelo
- **"usar"** o **"probar"** - Cómo usar el modelo
- **"páginas"** - Lista de páginas disponibles
- **"ayuda"** - Muestra este mensaje

¡Pregúntame cualquier cosa sobre el proyecto!
"""
    
    elif "precisión" in user_input_lower or "accuracy" in user_input_lower:
        if model_exists():
            model_info = get_model_info()
            if model_info:
                return f"""📈 **Métricas del Modelo:**

- **Precisión en entrenamiento**: {model_info['accuracy']*100:.2f}%
- **Precisión en validación**: {model_info['val_accuracy']*100:.2f}%
- **Pérdida en entrenamiento**: {model_info['loss']:.4f}
- **Pérdida en validación**: {model_info['val_loss']:.4f}

{'✅ Excelente rendimiento!' if model_info['val_accuracy'] > 0.9 else '⚠️ El modelo podría mejorarse con más entrenamiento.'}
"""
        return "⚠️ No hay un modelo entrenado todavía. Ve a la página de **Entrenamiento** para crear uno."
    
    else:
        return f"""🤔 Interesante pregunta: "{user_input}"

No estoy seguro de cómo responder específicamente a eso, pero puedo ayudarte con:

- Información sobre el **dataset EMNIST**
- Detalles del **modelo CNN**
- Cómo **entrenar** el modelo
- Cómo **usar** el sistema de reconocimiento
- Navegación por las **páginas**

Escribe **"ayuda"** para ver todos los comandos disponibles.
"""

# Contenedor de mensajes del chat
chat_container = st.container()

with chat_container:
    # Mostrar historial de mensajes usando session_state
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Input del chat
if prompt := st.chat_input("Pregúntame sobre el proyecto o los datos..."):
    # Añadir mensaje del usuario
    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    
    with chat_container:
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generar respuesta del asistente
        with st.chat_message("assistant"):
            response = generate_chat_response(prompt, dataset_info)
            st.markdown(response)
    
    # Añadir respuesta del asistente
    st.session_state.chat_messages.append({"role": "assistant", "content": response})
    st.rerun()

# Botón para limpiar chat
if st.session_state.chat_messages:
    if st.button("🗑️ Limpiar conversación"):
        st.session_state.chat_messages = []
        st.rerun()

# Footer
st.divider()
st.caption("""
💡 **Proyecto de Machine Learning con Streamlit**

Elementos integrados: 
✅ Visualización de datos | ✅ Chat bot | ✅ Widgets | ✅ Session State | ✅ Cache | ✅ Persistencia | ✅ Páginas múltiples
""")
