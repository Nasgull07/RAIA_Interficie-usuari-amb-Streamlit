"""
Módulo para carga y procesamiento de datos EMNIST
Elemento: CACHE DE DATOS - Los datos se cachean para evitar recargas
"""

import pandas as pd
import numpy as np
import streamlit as st
import os
from pathlib import Path

# Rutas a los archivos de datos
DOWNLOADS_PATH = Path.home() / "Downloads"
TRAIN_FILE = DOWNLOADS_PATH / "emnist-letters-train.csv" / "emnist-letters-train.csv"
TEST_FILE = DOWNLOADS_PATH / "emnist-letters-test.csv" / "emnist-letters-test.csv"

# Mapeo de índices a letras (EMNIST letters usa 1-26 para A-Z)
LETTER_MAPPING = {i: chr(64 + i) for i in range(1, 27)}  # 1->A, 2->B, ..., 26->Z

@st.cache_data(show_spinner="Cargando datos de entrenamiento...")
def load_train_data(sample_size=None):
    """
    Carga los datos de entrenamiento EMNIST
    Elemento: CACHE DE DATOS - Cachea los datos para evitar recargas
    
    Args:
        sample_size: Número de muestras a cargar (None para cargar todo)
    
    Returns:
        tuple: (X_train, y_train) donde X son las imágenes y y las etiquetas
    """
    try:
        if not TRAIN_FILE.exists():
            st.error(f"No se encontró el archivo: {TRAIN_FILE}")
            return None, None
        
        # Cargar dataset
        if sample_size:
            df = pd.read_csv(TRAIN_FILE, nrows=sample_size)
        else:
            df = pd.read_csv(TRAIN_FILE)
        
        # Separar etiquetas y características
        y = df.iloc[:, 0].values  # Primera columna son las etiquetas
        X = df.iloc[:, 1:].values  # Resto son los píxeles
        
        # Reshape a formato de imagen 28x28
        X = X.reshape(-1, 28, 28)
        
        # Normalizar píxeles a rango [0, 1]
        X = X.astype('float32') / 255.0
        
        return X, y
    
    except Exception as e:
        st.error(f"Error al cargar datos de entrenamiento: {str(e)}")
        return None, None

@st.cache_data(show_spinner="Cargando datos de prueba...")
def load_test_data(sample_size=None):
    """
    Carga los datos de prueba EMNIST
    Elemento: CACHE DE DATOS
    
    Args:
        sample_size: Número de muestras a cargar (None para cargar todo)
    
    Returns:
        tuple: (X_test, y_test)
    """
    try:
        if not TEST_FILE.exists():
            st.error(f"No se encontró el archivo: {TEST_FILE}")
            return None, None
        
        if sample_size:
            df = pd.read_csv(TEST_FILE, nrows=sample_size)
        else:
            df = pd.read_csv(TEST_FILE)
        
        y = df.iloc[:, 0].values
        X = df.iloc[:, 1:].values
        
        X = X.reshape(-1, 28, 28)
        X = X.astype('float32') / 255.0
        
        return X, y
    
    except Exception as e:
        st.error(f"Error al cargar datos de prueba: {str(e)}")
        return None, None

def label_to_letter(label):
    """
    Convierte una etiqueta numérica a letra
    
    Args:
        label: Número entre 1 y 26
    
    Returns:
        str: Letra correspondiente
    """
    return LETTER_MAPPING.get(label, '?')

def letter_to_label(letter):
    """
    Convierte una letra a etiqueta numérica
    
    Args:
        letter: Letra entre A y Z
    
    Returns:
        int: Etiqueta numérica
    """
    return ord(letter.upper()) - 64 if letter.isalpha() else 0

@st.cache_data
def get_dataset_info():
    """
    Obtiene información sobre los datasets
    Elemento: CACHE DE DATOS
    
    Returns:
        dict: Información de los datasets
    """
    info = {
        'train_file_exists': TRAIN_FILE.exists(),
        'test_file_exists': TEST_FILE.exists(),
        'train_path': str(TRAIN_FILE),
        'test_path': str(TEST_FILE),
        'num_classes': 26,
        'image_shape': (28, 28),
        'letters': [chr(i) for i in range(65, 91)]  # A-Z
    }
    
    # Intentar obtener tamaños de archivos
    if info['train_file_exists']:
        try:
            df = pd.read_csv(TRAIN_FILE, nrows=1)
            info['train_samples'] = len(pd.read_csv(TRAIN_FILE))
        except:
            info['train_samples'] = 'Desconocido'
    
    if info['test_file_exists']:
        try:
            info['test_samples'] = len(pd.read_csv(TEST_FILE))
        except:
            info['test_samples'] = 'Desconocido'
    
    return info

def preprocess_image_for_prediction(image):
    """
    Preprocesa una imagen para predicción
    
    Args:
        image: numpy array con la imagen
    
    Returns:
        numpy array: Imagen preprocesada
    """
    # Asegurar que es escala de grises
    if len(image.shape) == 3:
        image = np.mean(image, axis=2)
    
    # Redimensionar a 28x28 si es necesario
    if image.shape != (28, 28):
        from PIL import Image
        img_pil = Image.fromarray((image * 255).astype(np.uint8))
        img_pil = img_pil.resize((28, 28))
        image = np.array(img_pil) / 255.0
    
    # Normalizar
    image = image.astype('float32')
    if image.max() > 1.0:
        image = image / 255.0
    
    return image

def create_sample_images(num_samples=10):
    """
    Crea imágenes de muestra del dataset
    
    Args:
        num_samples: Número de muestras a obtener
    
    Returns:
        tuple: (imágenes, etiquetas)
    """
    X, y = load_test_data(sample_size=num_samples)
    return X, y

@st.cache_data
def get_class_distribution(y):
    """
    Calcula la distribución de clases
    Elemento: CACHE DE DATOS
    
    Args:
        y: Array de etiquetas
    
    Returns:
        dict: Distribución de clases
    """
    unique, counts = np.unique(y, return_counts=True)
    distribution = {}
    
    for label, count in zip(unique, counts):
        letter = label_to_letter(label)
        distribution[letter] = count
    
    return distribution
