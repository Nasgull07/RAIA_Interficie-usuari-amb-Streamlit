"""
Módulo para persistencia de datos entre sesiones
Elemento: PERSISTENCIA DE DATOS ENTRE SESIONES
Justificación: Guardar el historial de procesamiento para acceder a él incluso después de cerrar la aplicación
"""

import json
import os
from datetime import datetime
import streamlit as st

DATA_DIR = "data"
HISTORY_FILE = os.path.join(DATA_DIR, "ocr_history.json")

def ensure_data_directory():
    """
    Asegura que el directorio de datos existe
    """
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

@st.cache_data(ttl=60)  # Cache por 60 segundos
def load_history():
    """
    Carga el historial de procesamiento desde el archivo
    Elemento: CACHE DE DATOS + PERSISTENCIA
    
    Returns:
        list con el historial
    """
    ensure_data_directory()
    
    if not os.path.exists(HISTORY_FILE):
        return []
    
    try:
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            history = json.load(f)
        return history
    except Exception as e:
        print(f"Error al cargar historial: {e}")
        return []

def save_history(history):
    """
    Guarda el historial en el archivo
    Elemento: PERSISTENCIA DE DATOS
    
    Args:
        history: list con el historial a guardar
    """
    ensure_data_directory()
    
    try:
        # Guardar solo los últimos 100 registros para no sobrecargar
        history_to_save = history[-100:] if len(history) > 100 else history
        
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history_to_save, f, indent=2, ensure_ascii=False)
        
        # Limpiar cache para forzar recarga
        load_history.clear()
        
        return True
    except Exception as e:
        print(f"Error al guardar historial: {e}")
        return False

def export_history_to_csv(history):
    """
    Exporta el historial a formato CSV
    
    Args:
        history: list con el historial
    
    Returns:
        string con el contenido CSV
    """
    import csv
    import io
    
    output = io.StringIO()
    
    if not history:
        return ""
    
    # Obtener las claves del primer elemento
    fieldnames = ['timestamp', 'filename', 'word_count', 'confidence', 'text']
    
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    
    for entry in history:
        row = {
            'timestamp': entry.get('timestamp', ''),
            'filename': entry.get('filename', ''),
            'word_count': entry.get('word_count', 0),
            'confidence': entry.get('confidence', 0),
            'text': entry.get('text', '').replace('\n', ' ')[:200]  # Limitar texto
        }
        writer.writerow(row)
    
    return output.getvalue()

def get_storage_info():
    """
    Obtiene información sobre el almacenamiento
    
    Returns:
        dict con información del almacenamiento
    """
    ensure_data_directory()
    
    if not os.path.exists(HISTORY_FILE):
        return {
            'exists': False,
            'size': 0,
            'entries': 0
        }
    
    file_size = os.path.getsize(HISTORY_FILE)
    
    history = load_history()
    
    return {
        'exists': True,
        'size': file_size,
        'size_kb': round(file_size / 1024, 2),
        'entries': len(history),
        'last_modified': datetime.fromtimestamp(os.path.getmtime(HISTORY_FILE)).isoformat()
    }
