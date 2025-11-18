# Configuración del directorio de datos
DATA_DIR = "data"

# Configuración de Tesseract
# Descomenta y ajusta la siguiente línea si estás en Windows
# TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Idiomas soportados
SUPPORTED_LANGUAGES = {
    'spa': 'Español',
    'eng': 'English',
    'cat': 'Català',
    'fra': 'Français',
    'deu': 'Deutsch'
}

# Configuración de cache
CACHE_TTL_SECONDS = 60  # 1 minuto para load_history
STATS_CACHE_TTL = 300   # 5 minutos para estadísticas

# Límite de registros en el historial
MAX_HISTORY_RECORDS = 100
