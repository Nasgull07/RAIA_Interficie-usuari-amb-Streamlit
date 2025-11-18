"""
Módulo para análisis de texto
Proporciona estadísticas y análisis del texto extraído
"""

import re
from collections import Counter
import streamlit as st

@st.cache_data
def analyze_text(text):
    """
    Analiza el texto y retorna estadísticas detalladas
    Elemento: CACHE DE DATOS - cachea resultados del análisis
    
    Args:
        text: String con el texto a analizar
    
    Returns:
        dict con estadísticas del texto
    """
    if not text or text.strip() == "":
        return {
            'word_count': 0,
            'char_count': 0,
            'line_count': 0,
            'unique_words': 0,
            'avg_word_length': 0,
            'most_common_words': []
        }
    
    # Limpiar el texto
    lines = text.split('\n')
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Calcular estadísticas
    word_count = len(words)
    char_count = len(text)
    line_count = len([line for line in lines if line.strip()])
    unique_words = len(set(words))
    
    avg_word_length = sum(len(word) for word in words) / word_count if word_count > 0 else 0
    
    # Palabras más comunes
    word_freq = Counter(words)
    most_common = word_freq.most_common(10)
    
    return {
        'word_count': word_count,
        'char_count': char_count,
        'line_count': line_count,
        'unique_words': unique_words,
        'avg_word_length': round(avg_word_length, 2),
        'most_common_words': most_common
    }

def get_text_statistics(text):
    """
    Retorna estadísticas básicas del texto
    
    Args:
        text: String con el texto
    
    Returns:
        dict con estadísticas básicas
    """
    if not text or text.strip() == "":
        return {
            'word_count': 0,
            'char_count': 0,
            'line_count': 0,
            'avg_words_per_line': 0
        }
    
    lines = [line for line in text.split('\n') if line.strip()]
    words = text.split()
    
    word_count = len(words)
    line_count = len(lines)
    char_count = len(text)
    avg_words_per_line = word_count / line_count if line_count > 0 else 0
    
    return {
        'word_count': word_count,
        'char_count': char_count,
        'line_count': line_count,
        'avg_words_per_line': avg_words_per_line
    }

def search_in_text(text, search_term):
    """
    Busca un término en el texto y retorna las ocurrencias
    
    Args:
        text: String con el texto
        search_term: Término a buscar
    
    Returns:
        dict con resultados de la búsqueda
    """
    if not text or not search_term:
        return {
            'count': 0,
            'positions': []
        }
    
    text_lower = text.lower()
    search_lower = search_term.lower()
    
    count = text_lower.count(search_lower)
    
    # Encontrar posiciones
    positions = []
    start = 0
    while True:
        pos = text_lower.find(search_lower, start)
        if pos == -1:
            break
        positions.append(pos)
        start = pos + 1
    
    return {
        'count': count,
        'positions': positions
    }
