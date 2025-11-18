import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import plotly.graph_objects as go

st.set_page_config(page_title="Dibuja y Reconoce", page_icon="🎨", layout="wide")

from utils.model_builder import (
    load_model,
    predict_letter,
    model_exists,
    get_model_info,
    TF_AVAILABLE
)
from utils.data_loader import preprocess_image_for_prediction

st.title("🎨 Dibuja y Reconoce Letras")
st.markdown("""
Usa el modelo CNN entrenado para reconocer letras manuscritas. Puedes subir una imagen o usar imágenes de prueba.
""")

# Verificar que existe un modelo
if not model_exists():
    st.error("""
    ⚠️ **No hay modelo entrenado**
    
    Ve a la página **"🤖 Entrenamiento"** para entrenar un modelo primero.
    """)
    st.stop()

if not TF_AVAILABLE:
    st.error("""
    ⚠️ **TensorFlow no está instalado**
    
    Para usar esta funcionalidad, instala TensorFlow:
    ```bash
    pip install tensorflow
    ```
    """)
    st.stop()

# Cargar modelo
@st.cache_resource
def get_model():
    return load_model()

model = get_model()

if model is None:
    st.error("❌ Error al cargar el modelo")
    st.stop()

# Información del modelo en sidebar
with st.sidebar:
    st.header("🤖 Información del Modelo")
    model_info = get_model_info()
    
    if model_info:
        st.metric("Precisión", f"{model_info['val_accuracy']*100:.2f}%")
        st.metric("Épocas", model_info['epochs_trained'])
    
    st.divider()
    
    st.subheader("📊 Estadísticas de Sesión")
    if 'prediction_count' not in st.session_state:
        st.session_state.prediction_count = 0
    
    st.metric("Predicciones realizadas", st.session_state.prediction_count)

# Inicializar session_state
if 'current_image' not in st.session_state:
    st.session_state.current_image = None
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None

# Tabs para diferentes métodos de entrada
tab1, tab2, tab3 = st.tabs(["📤 Subir Imagen", "🖼️ Imágenes de Prueba", "📜 Historial"])

# TAB 1: Subir imagen
with tab1:
    st.header("Sube tu Imagen")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Cargar Imagen")
        
        uploaded_file = st.file_uploader(
            "Selecciona una imagen de una letra manuscrita",
            type=['png', 'jpg', 'jpeg', 'bmp'],
            help="La imagen debe contener una sola letra"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Imagen Original", use_container_width=True)
            
            if st.button("🔍 Reconocer Letra", type="primary", use_container_width=True):
                with st.spinner("Procesando..."):
                    # Preprocesar imagen
                    # Convertir a escala de grises
                    img_gray = ImageOps.grayscale(image)
                    
                    # Redimensionar a 28x28
                    img_resized = img_gray.resize((28, 28), Image.Resampling.LANCZOS)
                    
                    # Convertir a array y normalizar
                    img_array = np.array(img_resized).astype('float32') / 255.0
                    
                    # Invertir si es necesario (EMNIST espera fondo negro, letra blanca)
                    if img_array.mean() > 0.5:
                        img_array = 1.0 - img_array
                    
                    # Guardar en session state
                    st.session_state.current_image = img_array
                    
                    # Predecir
                    predicted_letter, probabilities = predict_letter(model, img_array)
                    
                    st.session_state.prediction_result = {
                        'letter': predicted_letter,
                        'probabilities': probabilities,
                        'image': img_array
                    }
                    
                    st.session_state.prediction_count += 1
                    
                    st.rerun()
    
    with col2:
        if st.session_state.prediction_result is not None:
            st.subheader("Resultado")
            
            result = st.session_state.prediction_result
            
            # Mostrar imagen preprocesada
            st.image(
                result['image'],
                caption="Imagen Procesada (28x28)",
                width=200
            )
            
            # Predicción principal
            st.markdown(f"## La letra es: **{result['letter']}**")
            
            # Confianza
            confidence = result['probabilities'][result['letter']] * 100
            st.progress(confidence / 100)
            st.caption(f"Confianza: {confidence:.2f}%")
            
            # Top 5 predicciones
            st.subheader("Top 5 Predicciones")
            
            sorted_probs = sorted(
                result['probabilities'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            
            for i, (letter, prob) in enumerate(sorted_probs, 1):
                col_a, col_b, col_c = st.columns([0.5, 2, 1])
                with col_a:
                    st.write(f"**{i}.**")
                with col_b:
                    st.write(f"**{letter}**")
                with col_c:
                    st.write(f"{prob*100:.2f}%")

# TAB 2: Imágenes de prueba
with tab2:
    st.header("Prueba con Imágenes del Dataset")
    
    from utils.data_loader import load_test_data, label_to_letter
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Selección")
        
        # Cargar algunas imágenes de prueba
        if 'test_images' not in st.session_state:
            with st.spinner("Cargando imágenes de prueba..."):
                X_test, y_test = load_test_data(sample_size=100)
                if X_test is not None:
                    st.session_state.test_images = (X_test, y_test)
        
        if 'test_images' in st.session_state:
            X_test, y_test = st.session_state.test_images
            
            # Selector de imagen
            img_idx = st.slider(
                "Selecciona una imagen",
                min_value=0,
                max_value=len(X_test)-1,
                value=0
            )
            
            if st.button("🎲 Imagen Aleatoria", use_container_width=True):
                img_idx = np.random.randint(0, len(X_test))
                st.rerun()
            
            # Mostrar imagen seleccionada
            st.image(X_test[img_idx], caption=f"Imagen #{img_idx}", width=200)
            
            true_label = label_to_letter(y_test[img_idx])
            st.info(f"**Etiqueta real**: {true_label}")
            
            if st.button("🔍 Predecir", type="primary", use_container_width=True):
                with st.spinner("Prediciendo..."):
                    predicted_letter, probabilities = predict_letter(model, X_test[img_idx])
                    
                    st.session_state.prediction_result = {
                        'letter': predicted_letter,
                        'probabilities': probabilities,
                        'image': X_test[img_idx],
                        'true_label': true_label
                    }
                    
                    st.session_state.prediction_count += 1
                    st.rerun()
    
    with col2:
        if st.session_state.prediction_result is not None and 'true_label' in st.session_state.prediction_result:
            result = st.session_state.prediction_result
            
            st.subheader("Resultado de la Predicción")
            
            # Comparar predicción con etiqueta real
            is_correct = result['letter'] == result['true_label']
            
            if is_correct:
                st.success(f"✅ **¡Correcto!** La letra es **{result['letter']}**")
            else:
                st.error(f"❌ **Incorrecto**. Predicho: **{result['letter']}**, Real: **{result['true_label']}**")
            
            # Gráfico de barras de probabilidades
            st.subheader("Distribución de Probabilidades")
            
            # Ordenar por probabilidad
            sorted_items = sorted(
                result['probabilities'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            letters = [item[0] for item in sorted_items]
            probs = [item[1] * 100 for item in sorted_items]
            
            # Colores: verde para correcto, rojo para predicción incorrecta
            colors = []
            for letter in letters:
                if letter == result['true_label']:
                    colors.append('green')
                elif letter == result['letter'] and not is_correct:
                    colors.append('red')
                else:
                    colors.append('lightblue')
            
            fig = go.Figure(data=[
                go.Bar(
                    x=letters,
                    y=probs,
                    marker_color=colors,
                    text=[f"{p:.1f}%" for p in probs],
                    textposition='outside'
                )
            ])
            
            fig.update_layout(
                title="Probabilidades para cada letra",
                xaxis_title="Letra",
                yaxis_title="Probabilidad (%)",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Leyenda
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("🟢 **Verde**: Etiqueta real")
            with col_b:
                if not is_correct:
                    st.markdown("🔴 **Rojo**: Predicción del modelo")

# TAB 3: Historial de predicciones
with tab3:
    st.header("📜 Historial de Predicciones")
    
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    
    if len(st.session_state.prediction_history) == 0:
        st.info("No hay predicciones en el historial aún. Realiza algunas predicciones primero.")
    else:
        st.write(f"Total de predicciones: **{len(st.session_state.prediction_history)}**")
        
        for i, pred in enumerate(reversed(st.session_state.prediction_history[-20:]), 1):
            with st.expander(f"Predicción #{len(st.session_state.prediction_history) - i + 1}: {pred['letter']}"):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.image(pred['image'], width=150)
                    st.metric("Letra Predicha", pred['letter'])
                    st.metric("Confianza", f"{pred['confidence']:.2f}%")
                
                with col2:
                    # Top 3 predicciones
                    st.write("**Top 3 probabilidades:**")
                    sorted_probs = sorted(
                        pred['probabilities'].items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:3]
                    
                    for letter, prob in sorted_probs:
                        st.write(f"- **{letter}**: {prob*100:.2f}%")

# Guardar predicción en historial
if st.session_state.prediction_result is not None:
    # Verificar si ya está en el historial
    if len(st.session_state.prediction_history) == 0 or \
       st.session_state.prediction_history[-1]['letter'] != st.session_state.prediction_result['letter']:
        
        result = st.session_state.prediction_result
        st.session_state.prediction_history.append({
            'letter': result['letter'],
            'confidence': result['probabilities'][result['letter']] * 100,
            'probabilities': result['probabilities'],
            'image': result['image']
        })

st.divider()
st.caption("""
💡 **Elementos integrados**:
- File uploader para cargar imágenes
- Sliders para navegar por imágenes de prueba
- Visualización interactiva con Plotly
- Session state para mantener historial de predicciones
- Cache de recursos para el modelo
""")
