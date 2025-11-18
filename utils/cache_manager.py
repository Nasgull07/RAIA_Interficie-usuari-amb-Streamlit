"""
Módulo para gestión de cache
Elemento: CACHE DE DATOS Y FUNCIONES
Justificación: Mejorar el rendimiento evitando reprocesar la misma imagen múltiples veces
"""

import streamlit as st
from PIL import Image
import hashlib
import io

def image_to_hash(image):
    """
    Genera un hash único para una imagen
    
    Args:
        image: PIL Image
    
    Returns:
        str hash de la imagen
    """
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    return hashlib.md5(img_byte_arr).hexdigest()

@st.cache_resource
def get_cached_model():
    """
    Cache de recursos para modelos o configuraciones pesadas
    Elemento: CACHE DE RECURSOS
    
    En un proyecto real, aquí cargarías modelos de ML pesados
    que no quieres recargar en cada interacción
    
    Returns:
        dict con recursos cacheados
    """
    return {
        'initialized': True,
        'model_name': 'Tesseract OCR',
        'version': '1.0'
    }

@st.cache_data(show_spinner=False)
def cache_ocr_result(image, language, psm, preprocessing):
    """
    Cachea los resultados del OCR para evitar reprocesar la misma imagen
    Elemento: CACHE DE DATOS
    
    Args:
        image: PIL Image
        language: Idioma del OCR
        psm: Page Segmentation Mode
        preprocessing: Bool si se aplica preprocesamiento
    
    Returns:
        dict con resultado del OCR
    """
    from utils.ocr_processor import process_image_ocr
    
    # Procesar la imagen
    result = process_image_ocr(image, language, psm, preprocessing)
    
    return result

@st.cache_data(ttl=300)  # Cache por 5 minutos
def get_processing_statistics(history):
    """
    Calcula estadísticas del historial con cache temporal
    Elemento: CACHE DE DATOS con TTL
    
    Args:
        history: list con el historial
    
    Returns:
        dict con estadísticas
    """
    if not history:
        return {
            'total_images': 0,
            'total_words': 0,
            'avg_words': 0,
            'avg_confidence': 0
        }
    
    total_images = len(history)
    total_words = sum(entry.get('word_count', 0) for entry in history)
    avg_words = total_words / total_images if total_images > 0 else 0
    
    confidences = [entry.get('confidence', 0) for entry in history]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    
    return {
        'total_images': total_images,
        'total_words': total_words,
        'avg_words': round(avg_words, 2),
        'avg_confidence': round(avg_confidence, 2)
    }

def clear_all_caches():
    """
    Limpia todos los caches de la aplicación
    """
    st.cache_data.clear()
    st.cache_resource.clear()
