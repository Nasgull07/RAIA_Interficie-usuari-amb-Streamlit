"""
Módulo para construcción y entrenamiento de modelos
Implementa CNN para reconocimiento de letras manuscritas
"""

import numpy as np
import streamlit as st
from pathlib import Path
import json
import pickle

# Intentar importar TensorFlow/Keras
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models
    from tensorflow.keras.utils import to_categorical
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    st.warning("⚠️ TensorFlow no está instalado. Instala con: pip install tensorflow")

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "letter_recognition_model.h5"
MODEL_INFO_PATH = MODEL_DIR / "model_info.json"

def create_cnn_model(input_shape=(28, 28, 1), num_classes=26):
    """
    Crea un modelo CNN para reconocimiento de letras
    Elemento: CACHE DE RECURSOS - El modelo se cachea en disco
    
    Args:
        input_shape: Forma de las imágenes de entrada
        num_classes: Número de clases (26 letras)
    
    Returns:
        keras.Model: Modelo compilado
    """
    if not TF_AVAILABLE:
        return None
    
    model = models.Sequential([
        # Primera capa convolucional
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Segunda capa convolucional
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Tercera capa convolucional
        layers.Conv2D(64, (3, 3), activation='relu'),
        
        # Capas densas
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compilar modelo
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(X_train, y_train, X_val, y_val, epochs=10, batch_size=128):
    """
    Entrena el modelo CNN
    
    Args:
        X_train: Datos de entrenamiento
        y_train: Etiquetas de entrenamiento
        X_val: Datos de validación
        y_val: Etiquetas de validación
        epochs: Número de épocas
        batch_size: Tamaño del batch
    
    Returns:
        tuple: (modelo, historial)
    """
    if not TF_AVAILABLE:
        return None, None
    
    # Preparar datos
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_val = X_val.reshape(-1, 28, 28, 1)
    
    # Ajustar etiquetas (EMNIST usa 1-26, necesitamos 0-25)
    y_train = y_train - 1
    y_val = y_val - 1
    
    # Crear modelo
    model = create_cnn_model()
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=0.00001
        )
    ]
    
    # Entrenar
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0
    )
    
    return model, history

@st.cache_resource
def load_model():
    """
    Carga el modelo guardado
    Elemento: CACHE DE RECURSOS - El modelo persiste en memoria
    
    Returns:
        keras.Model o None
    """
    if not TF_AVAILABLE:
        return None
    
    if MODEL_PATH.exists():
        try:
            model = keras.models.load_model(MODEL_PATH)
            return model
        except Exception as e:
            st.error(f"Error al cargar modelo: {str(e)}")
            return None
    else:
        return None

def save_model(model, history=None):
    """
    Guarda el modelo y su información
    Elemento: PERSISTENCIA DE DATOS
    
    Args:
        model: Modelo de Keras
        history: Historial de entrenamiento
    """
    MODEL_DIR.mkdir(exist_ok=True)
    
    # Guardar modelo
    model.save(MODEL_PATH)
    
    # Guardar información del modelo
    if history is not None:
        model_info = {
            'accuracy': float(history.history['accuracy'][-1]),
            'val_accuracy': float(history.history['val_accuracy'][-1]),
            'loss': float(history.history['loss'][-1]),
            'val_loss': float(history.history['val_loss'][-1]),
            'epochs_trained': len(history.history['accuracy'])
        }
        
        with open(MODEL_INFO_PATH, 'w') as f:
            json.dump(model_info, f, indent=2)
    
    # Limpiar cache para forzar recarga
    load_model.clear()

def get_model_info():
    """
    Obtiene información del modelo guardado
    
    Returns:
        dict con información del modelo
    """
    if MODEL_INFO_PATH.exists():
        with open(MODEL_INFO_PATH, 'r') as f:
            return json.load(f)
    return None

def predict_letter(model, image):
    """
    Predice la letra de una imagen
    
    Args:
        model: Modelo de Keras
        image: Imagen 28x28
    
    Returns:
        tuple: (letra_predicha, probabilidades)
    """
    if model is None:
        return None, None
    
    # Preparar imagen
    if len(image.shape) == 2:
        image = image.reshape(1, 28, 28, 1)
    elif len(image.shape) == 3 and image.shape[2] == 1:
        image = image.reshape(1, 28, 28, 1)
    
    # Predecir
    predictions = model.predict(image, verbose=0)
    predicted_class = np.argmax(predictions[0])
    
    # Convertir a letra (añadir 1 porque EMNIST usa 1-26)
    predicted_letter = chr(65 + predicted_class)  # A-Z
    
    # Probabilidades para todas las letras
    probabilities = {chr(65 + i): float(predictions[0][i]) for i in range(26)}
    
    return predicted_letter, probabilities

def model_exists():
    """
    Verifica si existe un modelo guardado
    
    Returns:
        bool
    """
    return MODEL_PATH.exists()
