# 🚀 Guía de Inicio Rápido - Proyecto OCR

## ⚡ Instalación Rápida

### 1. Instalar Tesseract OCR

**Windows:**
1. Descarga el instalador: https://github.com/UB-Mannheim/tesseract/wiki
2. Instala en la ruta por defecto: `C:\Program Files\Tesseract-OCR`
3. Añade idiomas durante la instalación (especialmente español)

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr tesseract-ocr-spa tesseract-ocr-cat
```

**macOS:**
```bash
brew install tesseract tesseract-lang
```

### 2. Instalar Dependencias de Python

```bash
pip install -r requirements.txt
```

### 3. Ejecutar la Aplicación

```bash
streamlit run app.py
```

## 🖼️ Imágenes de Prueba

Si no tienes imágenes de prueba, puedes crear algunas rápidamente:

1. Abre un documento de texto
2. Escribe algo de texto
3. Toma una captura de pantalla
4. Úsala en la aplicación

O busca imágenes de ejemplo en internet con texto claro.

## 🎯 Primeros Pasos

1. **Carga una imagen** en el tab "📤 Cargar Imagen"
2. **Haz clic** en "🔍 Extraer Texto"
3. **Explora** las diferentes tabs para ver todas las funcionalidades
4. **Prueba el chatbot** escribiendo "ayuda" en el tab "💬 Chat Asistente"

## ⚠️ Solución Rápida de Problemas

### Si Tesseract no se encuentra:

Edita el archivo `utils/ocr_processor.py` y descomenta/modifica esta línea:

```python
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

Ajusta la ruta según tu instalación.

### Si hay errores de importación:

```bash
pip install --upgrade streamlit pytesseract Pillow pandas plotly
```

## 📱 Características a Probar

- ✅ Prueba diferentes idiomas en el sidebar
- ✅ Activa/desactiva el preprocesamiento
- ✅ Procesa varias imágenes y ve el historial
- ✅ Explora las visualizaciones
- ✅ Chatea con el asistente
- ✅ Descarga los textos extraídos

¡Disfruta explorando el proyecto! 🎉
