# 🔤 Proyecto de Reconocimiento de Letras Manuscritas con Streamlit

## 📋 Descripción del Proyecto

Este proyecto implementa un **sistema completo de Machine Learning** para el reconocimiento de letras manuscritas utilizando el dataset **EMNIST Letters**. La aplicación web está construida con **Streamlit** e integra una **Red Neuronal Convolucional (CNN)** entrenada para clasificar letras del alfabeto (A-Z).

## 🎯 Características Principales

- ✅ **Visualización exploratoria de datos** del dataset EMNIST
- ✅ **Entrenamiento interactivo** de modelo CNN con configuración de hiperparámetros
- ✅ **Reconocimiento en tiempo real** de letras manuscritas
- ✅ **Sistema multi-página** con navegación fluida
- ✅ **Chatbot asistente** para consultas sobre el proyecto
- ✅ **Persistencia de modelos** entrenados
- ✅ **Visualizaciones interactivas** con Plotly
- ✅ **Caché optimizado** para rendimiento

## 🏗️ Estructura del Proyecto

```
RAIA_Interficie-usuari-amb-Streamlit/
│
├── app.py                                      # Aplicación principal (Inicio)
├── requirements.txt                            # Dependencias del proyecto
├── README.md                                   # Este archivo
├── config.py                                   # Configuración global
├── QUICKSTART.md                              # Guía rápida de inicio
│
├── data/                                      # Directorio para datos persistentes
│   └── (archivos generados automáticamente)
│
├── models/                                    # Modelos entrenados
│   ├── letter_recognition_model.h5           # Modelo CNN guardado
│   └── model_info.json                       # Información del modelo
│
├── pages/                                     # Páginas adicionales de Streamlit
│   ├── 1_📊_Visualizacion_Datos.py          # Exploración del dataset
│   ├── 2_🤖_Entrenamiento.py                 # Entrenamiento del modelo
│   └── 3_🎨_Dibuja_y_Reconoce.py            # Reconocimiento interactivo
│
└── utils/                                     # Módulos de utilidades
    ├── __init__.py
    ├── data_loader.py                        # Carga y procesamiento de datos EMNIST
    ├── model_builder.py                      # Construcción y entrenamiento CNN
    ├── text_analyzer.py                      # Análisis de texto (legacy)
    ├── ocr_processor.py                      # Procesamiento OCR (legacy)
    ├── persistence.py                        # Persistencia de datos
    └── cache_manager.py                      # Gestión de cache
```

## 📦 Instalación

### Requisitos Previos

1. **Python 3.8+**
2. **Datasets EMNIST** descargados en `~/Downloads/`:
   - `emnist-letters-train.csv`
   - `emnist-letters-test.csv`
   
   Puedes descargar los datasets desde: [EMNIST Dataset](https://www.nist.gov/itl/products-and-services/emnist-dataset)

### Instalación de Dependencias

```bash
# Clonar el repositorio
git clone https://github.com/Nasgull07/RAIA_Interficie-usuari-amb-Streamlit.git
cd RAIA_Interficie-usuari-amb-Streamlit

# Crear entorno virtual (opcional pero recomendado)
python -m venv .venv
.venv\Scripts\activate  # En Windows
# source .venv/bin/activate  # En Linux/Mac

# Instalar dependencias
pip install -r requirements.txt
```

## 🚀 Ejecución

```bash
python -m streamlit run app.py
```

La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`

## 📚 Guía de Uso

### 1. 🏠 Página Principal (Inicio)

La página de inicio proporciona:
- Información general del proyecto
- Estado del sistema (datasets y modelo)
- Demo rápida con imágenes de muestra
- **Chatbot asistente** para consultas interactivas

**Comandos del chatbot**:
- `dataset` - Información sobre el dataset EMNIST
- `modelo` - Detalles del modelo CNN
- `entrenar` - Cómo entrenar el modelo
- `usar` - Cómo usar el reconocimiento
- `ayuda` - Lista de comandos disponibles

### 2. 📊 Visualización de Datos

Explora el dataset EMNIST con múltiples visualizaciones:

**Características**:
- **Distribución de clases**: Gráfico de barras de frecuencia de letras
- **Galería de muestras**: Visualiza ejemplos de cada letra
- **Estadísticas avanzadas**: Análisis de intensidad y varianza de píxeles
- **Explorador interactivo**: Examina imágenes individuales en detalle

**Uso**:
1. Ajusta el tamaño de muestra en el sidebar (100-10,000)
2. Selecciona tipo de datos (Entrenamiento/Prueba)
3. Haz clic en "Cargar Datos"
4. Explora las diferentes tabs de visualización

### 3. 🤖 Entrenamiento del Modelo

Entrena una Red Neuronal Convolucional personalizada:

**Configuración disponible**:
- **Muestras de entrenamiento**: 1,000 - 100,000
- **Porcentaje de validación**: 10% - 30%
- **Épocas**: 1 - 20
- **Tamaño de batch**: 32, 64, 128, 256

**Proceso de entrenamiento**:
1. Configura los hiperparámetros en el sidebar
2. Haz clic en "Iniciar Entrenamiento"
3. Observa el progreso en tiempo real
4. Analiza las curvas de aprendizaje
5. El modelo se guarda automáticamente

**Métricas visualizadas**:
- Precisión de entrenamiento y validación
- Pérdida de entrenamiento y validación
- Curvas de aprendizaje interactivas
- Detección de overfitting

### 4. 🎨 Dibuja y Reconoce

Prueba el modelo entrenado:

**Métodos de entrada**:

**A) Subir Imagen**:
1. Sube una imagen de una letra manuscrita
2. Haz clic en "Reconocer Letra"
3. Visualiza la predicción y probabilidades

**B) Imágenes de Prueba**:
1. Usa el slider para seleccionar una imagen del dataset
2. Haz clic en "Predecir"
3. Compara la predicción con la etiqueta real
4. Visualiza gráfico de probabilidades

**C) Historial**:
- Revisa las últimas 20 predicciones realizadas
- Analiza confianza y probabilidades

## 📚 Elementos de Streamlit Implementados

### 1. 📊 VISUALIZACIÓN DE DATOS ✅

**Implementación**: 
- Gráficos interactivos con Plotly en página de Visualización
- Histogramas, gráficos de barras, gráficos de dispersión
- Matplotlib para análisis detallados

**Justificación**:
- Permite comprender la distribución del dataset antes del entrenamiento
- Identifica posibles desbalances en las clases
- Ayuda a detectar patrones y anomalías en los datos
- Visualiza el rendimiento del modelo durante entrenamiento

**Ubicación**: Página "Visualización de Datos", Página "Entrenamiento" (curvas de aprendizaje)

### 2. 💬 CHAT BOT ✅

**Implementación**: Chatbot conversacional en la página principal

**Justificación**:
- Proporciona una interfaz natural para obtener ayuda
- Responde preguntas sobre el proyecto, datasets y modelo
- Guía al usuario en el uso de la aplicación
- Mejora la accesibilidad y experiencia del usuario

**Funcionalidades**:
- Información del dataset
- Estado del modelo
- Guías de uso
- Recomendaciones
- Sistema de ayuda contextual

### 3. 🎛️ WIDGETS ✅

**Implementación**: Múltiples widgets en todas las páginas

**Widgets utilizados y su justificación**:

| Widget | Ubicación | Justificación |
|--------|-----------|---------------|
| `st.slider` | Visualización, Entrenamiento, Reconocimiento | Seleccionar rangos de valores (muestras, épocas, índices) |
| `st.number_input` | Entrenamiento | Entrada precisa de cantidades numéricas |
| `st.selectbox` | Visualización | Selección entre opciones predefinidas (tipo de datos) |
| `st.select_slider` | Entrenamiento | Selección de valores discretos (batch size) |
| `st.file_uploader` | Reconocimiento | Cargar imágenes para predicción |
| `st.button` | Todas las páginas | Ejecutar acciones (entrenar, predecir, cargar) |
| `st.checkbox` | (Disponible para expansiones) | Activar/desactivar opciones |
| `st.tabs` | Visualización, Reconocimiento | Organizar contenido relacionado |
| `st.expander` | Visualización, Reconocimiento | Mostrar información adicional de forma colapsable |
| `st.metric` | Todas las páginas | Mostrar KPIs y métricas clave |
| `st.progress` | Entrenamiento, Reconocimiento | Mostrar progreso de operaciones |
| `st.dataframe` | Visualización | Mostrar datos tabulares de forma interactiva |
| `st.chat_input` | Inicio | Interfaz de chat conversacional |
| `st.chat_message` | Inicio | Mostrar mensajes del chat |

### 4. 🔄 DEFINICIÓN DE ESTADO DE LA SESIÓN (Session State) ✅

**Implementación**: `st.session_state` para múltiples variables

**Justificación**:
- Mantiene datos entre reruns de la aplicación
- Esencial para el historial del chatbot
- Preserva resultados de entrenamiento
- Guarda predicciones realizadas
- Mantiene configuraciones del usuario

**Variables de estado utilizadas**:
```python
- prediction_history: Historial de predicciones realizadas
- model_loaded: Estado de carga del modelo
- current_prediction: Predicción actual mostrada
- chat_messages: Historial de conversación del chatbot
- sample_images: Imágenes de muestra cargadas
- viz_data: Datos de visualización cargados
- training_complete: Estado del entrenamiento
- training_history: Historial de métricas de entrenamiento
- test_images: Imágenes de prueba cargadas
- prediction_count: Contador de predicciones
```

### 5. 💾 CACHE DE DATOS/FUNCIONES ✅

**Implementación**: Decoradores `@st.cache_data` y `@st.cache_resource`

**Justificación**:
- **Mejora dramática de rendimiento**: Los datasets EMNIST son grandes
- **Evita recargas**: Los CSVs pueden tardar minutos en cargarse
- **Optimiza recursos**: El modelo CNN permanece en memoria
- **Experiencia fluida**: Navegación rápida entre páginas

**Funciones cacheadas**:

**`@st.cache_data`** (para datos):
- `load_train_data()`: Cachea dataset de entrenamiento
- `load_test_data()`: Cachea dataset de prueba
- `get_dataset_info()`: Cachea información del dataset
- `get_class_distribution()`: Cachea distribución de clases
- `prepare_history_dataframe()`: Cachea conversión a DataFrame

**`@st.cache_resource`** (para recursos):
- `load_model()`: Cachea el modelo CNN en memoria
- `get_tesseract_config()`: Cachea configuración (legacy)
- `get_model()`: Cachea instancia del modelo

### 6. 💿 PERSISTENCIA DE DATOS ENTRE SESIONES ✅

**Implementación**: 
- Archivos JSON para configuración
- Modelo H5 de Keras guardado en disco
- Sistema de archivos para datos persistentes

**Justificación**:
- **Continuidad**: El modelo entrenado no se pierde al cerrar la app
- **Reutilización**: Múltiples sesiones pueden usar el mismo modelo
- **Productividad**: No es necesario reentrenar en cada sesión
- **Compartir**: Los modelos pueden ser distribuidos fácilmente

**Mecanismo de persistencia**:

1. **Modelo entrenado**:
   - Archivo: `models/letter_recognition_model.h5`
   - Formato: Keras HDF5
   - Contiene: Arquitectura, pesos y configuración

2. **Información del modelo**:
   - Archivo: `models/model_info.json`
   - Contiene: Métricas, precisión, épocas

3. **Datos persistidos**:
```json
{
  "accuracy": 0.9234,
  "val_accuracy": 0.9156,
  "loss": 0.2145,
  "val_loss": 0.2389,
  "epochs_trained": 10
}
```

### 7. 📑 PÁGINAS MÚLTIPLES ✅

**Implementación**: Sistema de páginas de Streamlit

**Justificación**:
- **Organización**: Separa funcionalidades distintas
- **Escalabilidad**: Fácil añadir nuevas funcionalidades
- **Navegación intuitiva**: Menú lateral automático
- **Performance**: Carga solo lo necesario por página

**Páginas implementadas**:
1. **`app.py`** - Inicio (navegación, info, chatbot)
2. **`1_📊_Visualizacion_Datos.py`** - Exploración del dataset
3. **`2_🤖_Entrenamiento.py`** - Entrenamiento del modelo
4. **`3_🎨_Dibuja_y_Reconoce.py`** - Reconocimiento interactivo

## 🔧 Tecnologías Utilizadas

- **Streamlit** (1.28+): Framework de interfaz web
- **TensorFlow/Keras** (2.13+): Deep Learning y CNN
- **Pandas** (2.0+): Manipulación de datos
- **NumPy** (1.24+): Cálculo numérico
- **Plotly** (5.17+): Visualizaciones interactivas
- **Matplotlib** (3.7+): Gráficos estáticos
- **Pillow** (10.0+): Procesamiento de imágenes
- **Scikit-learn** (1.3+): Métricas y utilidades ML
- **Python** (3.8+): Lenguaje de programación

## 🎓 Arquitectura del Modelo CNN

```python
Model: Sequential
_________________________________________________________________
Layer (type)                Output Shape              Param #   
=================================================================
Conv2D (32 filters, 3x3)   (None, 26, 26, 32)        320       
MaxPooling2D (2x2)          (None, 13, 13, 32)        0         
Dropout (0.25)              (None, 13, 13, 32)        0         

Conv2D (64 filters, 3x3)   (None, 11, 11, 64)        18496     
MaxPooling2D (2x2)          (None, 5, 5, 64)          0         
Dropout (0.25)              (None, 5, 5, 64)          0         

Conv2D (64 filters, 3x3)   (None, 3, 3, 64)          36928     
Flatten                     (None, 576)               0         
Dense (128)                 (None, 128)               73856     
Dropout (0.5)               (None, 128)               0         
Dense (26, softmax)         (None, 26)                3354      
=================================================================
Total params: 132,954
```

**Características**:
- 3 capas convolucionales para extracción de características
- MaxPooling para reducción de dimensionalidad
- Dropout para regularización y prevenir overfitting
- Capa densa final con activación softmax para 26 clases

## 📊 Rendimiento Esperado

Con los parámetros recomendados:
- **Precisión en validación**: 85-92%
- **Tiempo de entrenamiento**: 5-15 minutos (10,000 muestras, 10 épocas)
- **Tamaño del modelo**: ~2 MB

## 🐛 Solución de Problemas

### Error: "Dataset not found"

**Solución**: Asegúrate de que los archivos EMNIST están en la ruta correcta:
```
~/Downloads/emnist-letters-train.csv/emnist-letters-train.csv
~/Downloads/emnist-letters-test.csv/emnist-letters-test.csv
```

### Error: "TensorFlow not installed"

**Solución**: 
```bash
pip install tensorflow
```

### Entrenamiento muy lento

**Soluciones**:
- Reduce el número de muestras de entrenamiento
- Disminuye el número de épocas
- Aumenta el batch size
- Verifica que estás usando GPU si está disponible

### Baja precisión del modelo

**Soluciones**:
- Aumenta el número de muestras de entrenamiento
- Incrementa el número de épocas
- Ajusta la tasa de aprendizaje
- Verifica la calidad de los datos de entrada

## 📖 Documentación Adicional

- **QUICKSTART.md**: Guía rápida de inicio
- **Comentarios en código**: Cada función está documentada
- **Docstrings**: Documentación completa de funciones
- **Tooltips**: Ayuda contextual en la interfaz

## 🎯 Casos de Uso

- **Educación**: Aprender Deep Learning y Computer Vision
- **Prototipado**: Probar arquitecturas de CNN
- **Demo**: Mostrar capacidades de reconocimiento de caracteres
- **Investigación**: Experimentar con hiperparámetros
- **Benchmarking**: Comparar rendimiento de modelos

## 🚧 Trabajo Futuro

Posibles mejoras:
- [ ] Implementar data augmentation
- [ ] Añadir más arquitecturas de modelos
- [ ] Exportar modelos a otros formatos (ONNX, TFLite)
- [ ] Implementar canvas de dibujo en tiempo real
- [ ] Añadir comparación de modelos
- [ ] Implementar transfer learning
- [ ] Añadir soporte para más datasets

## 📄 Licencia

Este proyecto es de código abierto y está disponible para uso educativo.

## 👨‍💻 Autor

Proyecto desarrollado como demostración completa de interfaz de usuario con Streamlit para Machine Learning, 
integrando todos los elementos del framework y un modelo real de Deep Learning.

## 🙏 Agradecimientos

- **Streamlit** por el excelente framework
- **TensorFlow/Keras** por las herramientas de Deep Learning
- **NIST** por el dataset EMNIST
- **La comunidad de código abierto**

---

## 📝 Justificación de Elementos (Resumen)

| Elemento | ¿Por qué se incluyó? |
|----------|---------------------|
| **Visualización de Datos** | Explorar y entender el dataset antes del entrenamiento; visualizar métricas de rendimiento |
| **Chat Bot** | Proporcionar ayuda contextual y mejorar la experiencia del usuario con interacción natural |
| **Widgets** | Permitir configuración interactiva de hiperparámetros y navegación intuitiva |
| **Session State** | Mantener estado entre reruns (historial, configuraciones, resultados) |
| **Cache** | Optimizar rendimiento evitando recargas de datos pesados y modelo |
| **Persistencia** | Guardar modelos entrenados entre sesiones para reutilización |
| **Páginas Múltiples** | Organizar funcionalidades distintas de forma escalable y clara |

---

**Nota**: Este es un proyecto educativo completo que demuestra las capacidades de Streamlit 
para crear aplicaciones de Machine Learning interactivas y profesionales.
