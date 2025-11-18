"""
Utilidad para segmentación y reconocimiento de texto completo
Detecta y segmenta caracteres individuales en imágenes de texto
"""

import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import cv2
import streamlit as st

def preprocess_text_image(image):
    """
    Preprocesa una imagen de texto para segmentación
    
    Args:
        image: PIL Image o numpy array
    
    Returns:
        numpy array: Imagen preprocesada
    """
    # Convertir a PIL si es necesario
    if isinstance(image, np.ndarray):
        image = Image.fromarray((image * 255).astype(np.uint8))
    
    # Convertir a escala de grises
    img_gray = ImageOps.grayscale(image)
    
    # Aumentar contraste
    enhancer = ImageEnhance.Contrast(img_gray)
    img_contrast = enhancer.enhance(2.5)
    
    # Aplicar filtro para reducir ruido
    img_filtered = img_contrast.filter(ImageFilter.MedianFilter(size=3))
    
    # Convertir a numpy array
    img_array = np.array(img_filtered)
    
    return img_array

def binarize_image(img_array, threshold='auto'):
    """
    Binariza la imagen (blanco y negro puro)
    
    Args:
        img_array: numpy array de la imagen
        threshold: umbral o 'auto' para Otsu
    
    Returns:
        numpy array: Imagen binarizada (texto blanco, fondo negro)
    """
    # Detectar si el fondo es claro u oscuro
    # Tomamos el promedio de los píxeles del borde (asumimos que es fondo)
    border_pixels = np.concatenate([
        img_array[0, :],      # borde superior
        img_array[-1, :],     # borde inferior
        img_array[:, 0],      # borde izquierdo
        img_array[:, -1]      # borde derecho
    ])
    avg_border = np.mean(border_pixels)
    
    # Si el promedio del borde es > 127, el fondo es claro (necesita inversión)
    needs_inversion = avg_border > 127
    
    if threshold == 'auto':
        # Método de Otsu para umbral automático
        try:
            if needs_inversion:
                # Fondo claro, texto oscuro -> invertir
                _, binary = cv2.threshold(
                    img_array,
                    0, 255,
                    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
                )
            else:
                # Fondo oscuro, texto claro -> no invertir
                _, binary = cv2.threshold(
                    img_array,
                    0, 255,
                    cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
        except:
            # Fallback si cv2 no está disponible
            threshold_val = np.mean(img_array)
            if needs_inversion:
                binary = (img_array < threshold_val).astype(np.uint8) * 255
            else:
                binary = (img_array > threshold_val).astype(np.uint8) * 255
    else:
        if needs_inversion:
            binary = (img_array < threshold).astype(np.uint8) * 255
        else:
            binary = (img_array > threshold).astype(np.uint8) * 255
    
    return binary

def find_text_lines(binary_img, min_line_height=10):
    """
    Encuentra las líneas de texto en la imagen
    
    Args:
        binary_img: Imagen binarizada
        min_line_height: Altura mínima de línea
    
    Returns:
        list: Lista de tuplas (y_start, y_end) para cada línea
    """
    # Proyección horizontal (suma de píxeles por fila)
    horizontal_projection = np.sum(binary_img, axis=1)
    
    # Encontrar zonas con texto
    in_text = False
    lines = []
    start_y = 0
    
    for i, val in enumerate(horizontal_projection):
        if val > 0 and not in_text:
            # Inicio de línea
            start_y = i
            in_text = True
        elif val == 0 and in_text:
            # Fin de línea
            if i - start_y >= min_line_height:
                lines.append((start_y, i))
            in_text = False
    
    # Si termina en texto
    if in_text and len(horizontal_projection) - start_y >= min_line_height:
        lines.append((start_y, len(horizontal_projection)))
    
    return lines

def find_characters_in_line(binary_line, min_width=5, min_height=10, merge_gap=3):
    """
    Encuentra caracteres individuales en una línea de texto
    
    Args:
        binary_line: Imagen binarizada de una línea
        min_width: Ancho mínimo de carácter
        min_height: Alto mínimo de carácter
        merge_gap: Píxeles de gap para fusionar caracteres fragmentados
    
    Returns:
        list: Lista de tuplas (x_start, x_end) para cada carácter
    """
    # Proyección vertical (suma de píxeles por columna)
    vertical_projection = np.sum(binary_line, axis=0)
    
    # Normalizar proyección para detectar valles
    if vertical_projection.max() > 0:
        normalized_proj = vertical_projection / vertical_projection.max()
    else:
        return []
    
    # Encontrar caracteres con fusión de gaps pequeños
    in_char = False
    chars = []
    start_x = 0
    gap_count = 0
    
    # Umbral para detectar espacios entre caracteres (más estricto)
    threshold_valley = 0.05  # 5% de la altura máxima para ser más sensible
    
    for i, val in enumerate(normalized_proj):
        if val > threshold_valley:
            if not in_char:
                # Inicio de carácter
                start_x = i
                in_char = True
                gap_count = 0
            else:
                gap_count = 0
        elif in_char:
            gap_count += 1
            # Solo terminar carácter si el gap es mayor que merge_gap
            if gap_count > merge_gap:
                # Fin de carácter
                if i - gap_count - start_x >= min_width:
                    chars.append((start_x, i - gap_count))
                in_char = False
                gap_count = 0
    
    # Si termina en carácter
    if in_char and len(normalized_proj) - start_x >= min_width:
        chars.append((start_x, len(normalized_proj)))
    
    # Si solo se detectó un carácter muy ancho, intentar dividirlo
    if len(chars) == 1 and (chars[0][1] - chars[0][0]) > min_width * 2.5:
        x_start, x_end = chars[0]
        char_projection = normalized_proj[x_start:x_end]
        
        # Buscar valles (mínimos locales) significativos
        valleys = []
        for i in range(1, len(char_projection) - 1):
            if (char_projection[i] < char_projection[i-1] and 
                char_projection[i] < char_projection[i+1] and
                char_projection[i] < 0.3):  # Valle más profundo
                valleys.append(x_start + i)
        
        # Dividir por los valles encontrados
        if valleys:
            new_chars = []
            prev = x_start
            for valley in valleys:
                if valley - prev >= min_width:
                    new_chars.append((prev, valley))
                    prev = valley
            if x_end - prev >= min_width:
                new_chars.append((prev, x_end))
            
            if len(new_chars) > 1:
                chars = new_chars
    
    return chars

def segment_text_image(image, min_char_width=5, min_char_height=10):
    """
    Segmenta una imagen de texto en caracteres individuales
    
    Args:
        image: PIL Image o numpy array
        min_char_width: Ancho mínimo de carácter
        min_char_height: Alto mínimo de carácter
    
    Returns:
        list: Lista de diccionarios con información de cada carácter
    """
    # Preprocesar
    img_array = preprocess_text_image(image)
    
    # Binarizar
    binary = binarize_image(img_array)
    
    # Encontrar líneas
    lines = find_text_lines(binary, min_line_height=min_char_height)
    
    all_chars = []
    
    # Para cada línea, encontrar caracteres
    for line_idx, (y_start, y_end) in enumerate(lines):
        line_img = binary[y_start:y_end, :]
        
        # Encontrar caracteres en la línea
        chars = find_characters_in_line(line_img, min_width=min_char_width)
        
        for char_idx, (x_start, x_end) in enumerate(chars):
            # Extraer carácter
            char_img = binary[y_start:y_end, x_start:x_end]
            
            # Añadir padding para hacer cuadrado con margen
            h, w = char_img.shape
            
            # Añadir 20% de padding alrededor
            padding = int(max(h, w) * 0.2)
            max_dim = max(h, w) + (padding * 2)
            
            # Crear imagen cuadrada con padding (fondo negro)
            padded = np.zeros((max_dim, max_dim), dtype=np.uint8)
            
            # Centrar el carácter
            y_offset = (max_dim - h) // 2
            x_offset = (max_dim - w) // 2
            padded[y_offset:y_offset+h, x_offset:x_offset+w] = char_img
            
            # Redimensionar a 28x28 con mejor interpolación
            char_pil = Image.fromarray(padded)
            char_resized = char_pil.resize((28, 28), Image.Resampling.LANCZOS)
            char_array = np.array(char_resized).astype('float32') / 255.0
            
            # IMPORTANTE: Normalizar para que coincida con EMNIST
            # EMNIST tiene letras BLANCAS en fondo NEGRO con valores normalizados
            # Invertir si el fondo es claro (media > 0.5)
            if char_array.mean() > 0.5:
                char_array = 1.0 - char_array
            
            all_chars.append({
                'image': char_array,
                'position': (line_idx, char_idx),
                'bbox': (x_start, y_start, x_end, y_end),
                'line': line_idx
            })
    
    return all_chars, binary

def recognize_text_from_image(image, model):
    """
    Reconoce texto completo de una imagen
    
    Args:
        image: PIL Image
        model: Modelo CNN entrenado
    
    Returns:
        tuple: (texto reconocido, caracteres segmentados, imagen procesada)
    """
    from utils.model_builder import predict_letter
    
    # Segmentar caracteres
    chars, binary_img = segment_text_image(image)
    
    if not chars:
        return "", [], binary_img
    
    # Reconocer cada carácter
    recognized_text = []
    current_line = -1
    
    for char_info in chars:
        # Detectar cambio de línea
        if char_info['line'] != current_line:
            if current_line >= 0:
                recognized_text.append('\n')
            current_line = char_info['line']
        
        # Predecir letra
        predicted_letter, probabilities = predict_letter(model, char_info['image'])
        confidence = probabilities[predicted_letter]
        
        # Añadir información de predicción
        char_info['predicted_letter'] = predicted_letter
        char_info['confidence'] = confidence
        char_info['probabilities'] = probabilities
        
        recognized_text.append(predicted_letter)
    
    # Unir texto
    text = ''.join(recognized_text)
    
    return text, chars, binary_img

@st.cache_data
def get_text_statistics(text):
    """
    Calcula estadísticas del texto reconocido
    
    Args:
        text: String de texto
    
    Returns:
        dict: Estadísticas
    """
    lines = text.split('\n')
    words = text.replace('\n', ' ').split()
    
    return {
        'total_chars': len(text.replace('\n', '')),
        'total_lines': len(lines),
        'total_words': len(words),
        'chars_per_line': len(text.replace('\n', '')) / max(len(lines), 1),
        'words_per_line': len(words) / max(len(lines), 1)
    }
