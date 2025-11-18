import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="Entrenamiento del Modelo", page_icon="🤖", layout="wide")

from utils.data_loader import load_train_data, load_test_data
from utils.model_builder import (
    train_model,
    save_model,
    load_model,
    model_exists,
    get_model_info,
    TF_AVAILABLE
)

st.title("🤖 Entrenamiento del Modelo CNN")
st.markdown("""
Entrena una Red Neuronal Convolucional para reconocer letras manuscritas del dataset EMNIST.
""")

# Verificar TensorFlow
if not TF_AVAILABLE:
    st.error("""
    ⚠️ **TensorFlow no está instalado**
    
    Para usar esta funcionalidad, instala TensorFlow:
    ```bash
    pip install tensorflow
    ```
    """)
    st.stop()

# Sidebar con configuración de entrenamiento
with st.sidebar:
    st.header("⚙️ Configuración de Entrenamiento")
    
    # Widgets para hiperparámetros
    st.subheader("Datos")
    train_samples = st.number_input(
        "Muestras de entrenamiento",
        min_value=1000,
        max_value=100000,
        value=10000,
        step=1000,
        help="Número de muestras para entrenar"
    )
    
    val_split = st.slider(
        "% Validación",
        min_value=10,
        max_value=30,
        value=20,
        help="Porcentaje de datos para validación"
    )
    
    st.subheader("Hiperparámetros")
    epochs = st.slider(
        "Épocas",
        min_value=1,
        max_value=20,
        value=10,
        help="Número de épocas de entrenamiento"
    )
    
    batch_size = st.select_slider(
        "Tamaño de batch",
        options=[32, 64, 128, 256],
        value=128,
        help="Tamaño del batch"
    )
    
    st.divider()
    
    # Estado del modelo actual
    st.subheader("📊 Modelo Actual")
    if model_exists():
        st.success("✅ Modelo existente encontrado")
        model_info = get_model_info()
        if model_info:
            st.metric("Precisión", f"{model_info['val_accuracy']*100:.2f}%")
            st.metric("Épocas previas", model_info['epochs_trained'])
        
        st.warning("⚠️ Entrenar sobrescribirá el modelo actual")
    else:
        st.info("ℹ️ No hay modelo previo")

# Inicializar session_state
if 'training_complete' not in st.session_state:
    st.session_state.training_complete = False
if 'training_history' not in st.session_state:
    st.session_state.training_history = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = None

# Botón de entrenamiento
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    if st.button("🚀 Iniciar Entrenamiento", type="primary", use_container_width=True):
        st.session_state.training_complete = False
        st.session_state.training_history = None
        
        # Contenedor para progreso
        progress_container = st.container()
        
        with progress_container:
            # Cargar datos
            st.info("📥 Cargando datos...")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Cargando dataset de entrenamiento...")
            X_train_full, y_train_full = load_train_data(sample_size=train_samples)
            progress_bar.progress(20)
            
            if X_train_full is None or y_train_full is None:
                st.error("❌ Error al cargar los datos")
                st.stop()
            
            # Dividir en train y validation
            status_text.text("Preparando datos de validación...")
            val_size = int(len(X_train_full) * (val_split / 100))
            indices = np.random.permutation(len(X_train_full))
            
            train_idx = indices[val_size:]
            val_idx = indices[:val_size]
            
            X_train = X_train_full[train_idx]
            y_train = y_train_full[train_idx]
            X_val = X_train_full[val_idx]
            y_val = y_train_full[val_idx]
            
            progress_bar.progress(40)
            
            st.success(f"✅ Datos cargados: {len(X_train)} entrenamiento, {len(X_val)} validación")
            
            # Entrenar modelo
            status_text.text("🏋️ Entrenando modelo CNN...")
            
            # Crear contenedor para métricas en tiempo real
            metrics_container = st.empty()
            
            model, history = train_model(
                X_train, y_train,
                X_val, y_val,
                epochs=epochs,
                batch_size=batch_size
            )
            
            progress_bar.progress(90)
            
            if model is not None and history is not None:
                # Guardar modelo
                status_text.text("💾 Guardando modelo...")
                save_model(model, history)
                progress_bar.progress(100)
                
                st.session_state.training_complete = True
                st.session_state.training_history = history
                st.session_state.model_trained = model
                
                status_text.empty()
                st.success("✅ ¡Entrenamiento completado con éxito!")
                st.balloons()
                
                st.rerun()
            else:
                st.error("❌ Error durante el entrenamiento")

# Mostrar resultados si el entrenamiento está completo
if st.session_state.training_complete and st.session_state.training_history is not None:
    st.divider()
    st.header("📊 Resultados del Entrenamiento")
    
    history = st.session_state.training_history
    
    # Métricas finales
    col1, col2, col3, col4 = st.columns(4)
    
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    
    col1.metric("Precisión Entrenamiento", f"{final_train_acc*100:.2f}%")
    col2.metric("Precisión Validación", f"{final_val_acc*100:.2f}%")
    col3.metric("Pérdida Entrenamiento", f"{final_train_loss:.4f}")
    col4.metric("Pérdida Validación", f"{final_val_loss:.4f}")
    
    # Gráficos de entrenamiento
    st.subheader("📈 Curvas de Aprendizaje")
    
    tab1, tab2 = st.tabs(["Precisión", "Pérdida"])
    
    with tab1:
        # Gráfico de precisión
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            y=history.history['accuracy'],
            name='Entrenamiento',
            mode='lines+markers',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            y=history.history['val_accuracy'],
            name='Validación',
            mode='lines+markers',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title='Precisión durante el Entrenamiento',
            xaxis_title='Época',
            yaxis_title='Precisión',
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Gráfico de pérdida
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            y=history.history['loss'],
            name='Entrenamiento',
            mode='lines+markers',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            y=history.history['val_loss'],
            name='Validación',
            mode='lines+markers',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title='Pérdida durante el Entrenamiento',
            xaxis_title='Época',
            yaxis_title='Pérdida',
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Análisis del entrenamiento
    st.subheader("🔍 Análisis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Mejora por época
        improvements = []
        for i in range(1, len(history.history['val_accuracy'])):
            imp = history.history['val_accuracy'][i] - history.history['val_accuracy'][i-1]
            improvements.append(imp)
        
        if improvements:
            avg_improvement = np.mean(improvements)
            st.metric(
                "Mejora promedio por época",
                f"{avg_improvement*100:.2f}%",
                delta=f"{'Positiva' if avg_improvement > 0 else 'Negativa'}"
            )
    
    with col2:
        # Overfitting check
        overfit_gap = final_train_acc - final_val_acc
        st.metric(
            "Gap Entrenamiento-Validación",
            f"{overfit_gap*100:.2f}%",
            help="Diferencia entre precisión de entrenamiento y validación"
        )
        
        if overfit_gap > 0.1:
            st.warning("⚠️ Posible overfitting detectado")
        elif overfit_gap < 0.05:
            st.success("✅ Buen balance entre entrenamiento y validación")
    
    # Recomendaciones
    st.subheader("💡 Recomendaciones")
    
    if final_val_acc < 0.7:
        st.info("""
        📚 **Baja precisión detectada**
        - Aumenta el número de muestras de entrenamiento
        - Incrementa el número de épocas
        - Considera ajustar la arquitectura del modelo
        """)
    elif final_val_acc >= 0.7 and final_val_acc < 0.85:
        st.success("""
        👍 **Rendimiento aceptable**
        - El modelo funciona bien para casos básicos
        - Puedes mejorar incrementando épocas o datos
        """)
    else:
        st.success("""
        🎉 **Excelente rendimiento**
        - El modelo está listo para producción
        - Prueba el modelo en la página "Dibuja y Reconoce"
        """)

# Modelo actual
elif model_exists():
    st.info("""
    ℹ️ **Hay un modelo previamente entrenado**
    
    Puedes:
    - Entrenar un nuevo modelo (sobrescribirá el actual)
    - Ir a la página "Dibuja y Reconoce" para probarlo
    """)
    
    st.subheader("📊 Información del Modelo Actual")
    model_info = get_model_info()
    
    if model_info:
        col1, col2, col3 = st.columns(3)
        col1.metric("Precisión en Validación", f"{model_info['val_accuracy']*100:.2f}%")
        col2.metric("Épocas Entrenadas", model_info['epochs_trained'])
        col3.metric("Pérdida Final", f"{model_info['val_loss']:.4f}")

else:
    st.info("""
    👆 **Configura los parámetros y comienza el entrenamiento**
    
    Ajusta los hiperparámetros en el panel lateral y haz clic en "Iniciar Entrenamiento".
    
    **Recomendaciones iniciales:**
    - Comienza con 5,000-10,000 muestras para pruebas rápidas
    - Usa 5-10 épocas para el primer entrenamiento
    - Una vez satisfecho, entrena con más datos y épocas
    """)

st.divider()
st.caption("""
💡 **Elementos integrados**: 
- Widgets (sliders, number_input, select_slider) para configuración
- Progress bars y spinners para feedback visual
- Cache de datos para optimización
- Session state para mantener resultados del entrenamiento
- Persistencia del modelo en disco
""")
