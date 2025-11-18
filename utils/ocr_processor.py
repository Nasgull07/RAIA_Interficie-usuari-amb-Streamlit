"""
Módulo para procesamiento de OCR con cache
Implementa funciones cacheadas para mejorar el rendimiento
"""

import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import streamlit as st
import numpy as np

@st.cache_resource
def get_tesseract_config():
    """
    Cache del modelo/configuración de Tesseract
    Elemento: CACHE DE RECURSOS - persiste entre reruns
    """
    # Configuración base de Tesseract
    # En Windows, es posible que necesites especificar la ruta
    # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    return {
        'available': True,
        'version': 'Tesseract OCR'
    }

def preprocess_image(image, enhance_contrast=True, enhance_sharpness=True, binarize=True):
    """
    Preprocesa la imagen para mejorar el OCR
    
    Args:
        image: PIL Image
        enhance_contrast: Mejorar contraste
        enhance_sharpness: Mejorar nitidez
        binarize: Binarizar imagen
    
    Returns:
        PIL Image procesada
    """
    # Convertir a escala de grises
    img = image.convert('L')
    
    # Mejorar contraste
    if enhance_contrast:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2.0)
    
    # Mejorar nitidez
    if enhance_sharpness:
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(2.0)
    
    # Binarización (convertir a blanco y negro)
    if binarize:
        # Umbral adaptativo
        threshold = 128
        img = img.point(lambda x: 255 if x > threshold else 0)
    
    return img

def process_image_ocr(image, language='spa', psm=3, preprocess=True):
    """
    Procesa una imagen con OCR
    
    Args:
        image: PIL Image
        language: Código de idioma para Tesseract
        psm: Page Segmentation Mode
        preprocess: Aplicar preprocesamiento
    
    Returns:
        dict con 'text' y 'confidence'
    """
    try:
        # Aplicar preprocesamiento si está activado
        if preprocess:
            processed_image = preprocess_image(image)
        else:
            processed_image = image
        
        # Configurar Tesseract
        custom_config = f'--oem 3 --psm {psm} -l {language}'
        
        # Extraer texto
        text = pytesseract.image_to_string(processed_image, config=custom_config)
        
        # Obtener información de confianza
        try:
            data = pytesseract.image_to_data(processed_image, config=custom_config, output_type=pytesseract.Output.DICT)
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        except:
            avg_confidence = 0
        
        return {
            'text': text.strip(),
            'confidence': round(avg_confidence, 2)
        }
    
    except Exception as e:
        return {
            'text': f"Error al procesar la imagen: {str(e)}",
            'confidence': 0
        }
