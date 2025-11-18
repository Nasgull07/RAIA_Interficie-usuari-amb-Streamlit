import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import plotly.graph_objects as go
import matplotlib.pyplot as plt

st.set_page_config(page_title="Reconocimiento de Texto", page_icon="📝", layout="wide")

from utils.model_builder import load_model, model_exists, TF_AVAILABLE
from utils.text_segmentation import (
    recognize_text_from_image,
    segment_text_image,
    get_text_statistics
)

st.title("📝 Reconocimiento de Texto Completo")
st.markdown("""
Sube una imagen con una **frase o texto** y el sistema:
1. 🔍 **Segmenta** automáticamente cada letra
2. 🤖 **Reconoce** cada carácter con el modelo CNN
3. 📄 **Reconstruye** el texto completo
""")

# Verificar que existe un modelo
if not model_exists():
    st.error("""
    ⚠️ **No hay modelo entrenado**
    
    Ve a la página **"🤖 Entrenamiento"** para entrenar un modelo primero.
    """)
    st.stop()

if not TF_AVAILABLE:
    st.error("""
    ⚠️ **TensorFlow no está instalado**
    
    Para usar esta funcionalidad, instala TensorFlow:
    ```bash
    pip install tensorflow opencv-python
    ```
    """)
    st.stop()

# Cargar modelo
@st.cache_resource
def get_model():
    return load_model()

model = get_model()

if model is None:
    st.error("❌ Error al cargar el modelo")
    st.stop()

# Sidebar con configuración
with st.sidebar:
    st.header("⚙️ Configuración de Segmentación")
    
    st.subheader("Parámetros")
    
    min_char_width = st.slider(
        "Ancho mínimo de carácter",
        min_value=3,
        max_value=15,
        value=5,
        help="Píxeles mínimos de ancho para considerar un carácter"
    )
    
    min_char_height = st.slider(
        "Alto mínimo de carácter",
        min_value=5,
        max_value=20,
        value=10,
        help="Píxeles mínimos de alto para considerar un carácter"
    )
    
    confidence_threshold = st.slider(
        "Umbral de confianza (%)",
        min_value=0,
        max_value=100,
        value=30,
        help="Marcará en rojo caracteres con confianza menor a este valor"
    )
    
    st.divider()
    
    st.subheader("💡 Consejos")
    st.info("""
    **Para mejores resultados:**
    - Usa texto impreso o manuscrito claro
    - Letras bien separadas
    - Fondo claro, texto oscuro
    - Buena iluminación
    - Sin inclinación excesiva
    """)

# Inicializar session_state
if 'text_result' not in st.session_state:
    st.session_state.text_result = None
if 'recognition_history' not in st.session_state:
    st.session_state.recognition_history = []

# Tabs principales
tab1, tab2, tab3 = st.tabs(["📤 Reconocer Texto", "🖼️ Crear Texto de Prueba", "📜 Historial"])

# TAB 1: Reconocimiento
with tab1:
    st.header("Sube una Imagen con Texto")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Imagen Original")
        
        uploaded_file = st.file_uploader(
            "Selecciona una imagen con texto",
            type=['png', 'jpg', 'jpeg', 'bmp'],
            help="La imagen debe contener texto claro y legible"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Imagen Original", use_container_width=True)
            
            if st.button("🔍 Reconocer Texto", type="primary", use_container_width=True):
                with st.spinner("Procesando imagen..."):
                    # Reconocer texto
                    text, chars, binary_img = recognize_text_from_image(image, model)
                    
                    st.session_state.text_result = {
                        'text': text,
                        'chars': chars,
                        'binary_img': binary_img,
                        'original_img': image,
                        'num_chars': len(chars)
                    }
                    
                    # Añadir al historial
                    stats = get_text_statistics(text)
                    st.session_state.recognition_history.append({
                        'text': text,
                        'num_chars': len(chars),
                        'stats': stats
                    })
                    
                    st.success(f"✅ Texto reconocido! {len(chars)} caracteres detectados")
                    st.rerun()
    
    with col2:
        if st.session_state.text_result is not None:
            result = st.session_state.text_result
            
            st.subheader("Resultado")
            
            # Texto reconocido
            st.markdown("### 📄 Texto Reconocido:")
            st.text_area(
                "Texto",
                value=result['text'],
                height=150,
                help="Texto reconocido - puedes copiarlo"
            )
            
            # Botón de descarga
            st.download_button(
                label="📥 Descargar Texto",
                data=result['text'],
                file_name="texto_reconocido.txt",
                mime="text/plain",
                use_container_width=True
            )
            
            # Estadísticas
            st.subheader("📊 Estadísticas")
            stats = get_text_statistics(result['text'])
            
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Caracteres", stats['total_chars'])
            col_b.metric("Líneas", stats['total_lines'])
            col_c.metric("Palabras", stats['total_words'])
    
    # Visualización detallada
    if st.session_state.text_result is not None:
        st.divider()
        st.header("🔍 Análisis Detallado")
        
        result = st.session_state.text_result
        
        tab_a, tab_b, tab_c, tab_d = st.tabs(["Imagen Procesada", "Caracteres Detectados", "Confianza", "🔬 Primeras 10 Letras"])
        
        with tab_a:
            st.subheader("Imagen Binarizada")
            st.image(result['binary_img'], caption="Procesamiento para segmentación", use_container_width=True)
            
            st.info("""
            Esta imagen muestra cómo el sistema procesa el texto:
            - Blanco: caracteres detectados
            - Negro: fondo
            """)
        
        with tab_b:
            st.subheader(f"Caracteres Segmentados ({len(result['chars'])} encontrados)")
            
            # Mostrar cada carácter
            chars_per_row = 10
            rows = (len(result['chars']) + chars_per_row - 1) // chars_per_row
            
            for row in range(rows):
                cols = st.columns(chars_per_row)
                for col_idx in range(chars_per_row):
                    char_idx = row * chars_per_row + col_idx
                    if char_idx < len(result['chars']):
                        char_info = result['chars'][char_idx]
                        with cols[col_idx]:
                            # Color según confianza
                            confidence_pct = char_info['confidence'] * 100
                            if confidence_pct >= confidence_threshold:
                                st.image(char_info['image'], width=50)
                                st.caption(f"**{char_info['predicted_letter']}**")
                            else:
                                st.image(char_info['image'], width=50)
                                st.caption(f"🔴 **{char_info['predicted_letter']}**")
                            st.caption(f"{confidence_pct:.0f}%")
        
        with tab_c:
            st.subheader("Análisis de Confianza")
            
            # Preparar datos
            letters = [char['predicted_letter'] for char in result['chars']]
            confidences = [char['confidence'] * 100 for char in result['chars']]
            positions = list(range(len(letters)))
            
            # Gráfico de confianza
            fig = go.Figure()
            
            # Colores según umbral
            colors = ['red' if c < confidence_threshold else 'green' for c in confidences]
            
            fig.add_trace(go.Scatter(
                x=positions,
                y=confidences,
                mode='lines+markers',
                marker=dict(color=colors, size=8),
                text=letters,
                hovertemplate='<b>%{text}</b><br>Confianza: %{y:.1f}%<extra></extra>',
                line=dict(color='lightblue', width=2)
            ))
            
            # Línea de umbral
            fig.add_hline(
                y=confidence_threshold,
                line_dash="dash",
                line_color="orange",
                annotation_text=f"Umbral: {confidence_threshold}%"
            )
            
            fig.update_layout(
                title="Confianza por Carácter",
                xaxis_title="Posición del carácter",
                yaxis_title="Confianza (%)",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Estadísticas de confianza
            col1, col2, col3 = st.columns(3)
            col1.metric("Confianza Media", f"{np.mean(confidences):.1f}%")
            if len(confidences) > 0:
                col2.metric("Confianza Mínima", f"{np.min(confidences):.1f}%")
                col3.metric("Confianza Máxima", f"{np.max(confidences):.1f}%")
            else:
                col2.metric("Confianza Mínima", "N/A")
                col3.metric("Confianza Máxima", "N/A")
            
            # Advertencias
            low_confidence = sum(1 for c in confidences if c < confidence_threshold)
            if low_confidence > 0:
                st.warning(f"⚠️ {low_confidence} caracteres con baja confianza (< {confidence_threshold}%). Revisa el texto reconocido.")
        
        with tab_d:
            st.subheader("🔬 Verificación de Segmentación - Primeras 10 Letras")
            st.info("Aquí puedes ver cómo el sistema detectó y normalizó las primeras letras antes de reconocerlas")
            
            # Mostrar las primeras 10 letras detectadas
            num_to_show = min(10, len(result['chars']))
            
            if num_to_show > 0:
                st.markdown(f"**Mostrando las primeras {num_to_show} letras detectadas:**")
                
                # Crear una imagen compuesta con las 10 primeras letras
                char_size = 28
                spacing = 10
                composite_width = num_to_show * (char_size + spacing)
                composite_height = char_size + 60  # Espacio para texto abajo
                
                # Crear imagen compuesta
                composite = Image.new('RGB', (composite_width, composite_height), color=(240, 240, 240))
                draw = ImageDraw.Draw(composite)
                
                try:
                    font = ImageFont.truetype("arial.ttf", 12)
                except:
                    font = ImageFont.load_default()
                
                for i in range(num_to_show):
                    char_info = result['chars'][i]
                    x_pos = i * (char_size + spacing)
                    
                    # Pegar imagen del carácter
                    char_img = char_info['image']
                    # Convertir numpy array a PIL Image si es necesario
                    if isinstance(char_img, np.ndarray):
                        char_img = Image.fromarray((char_img * 255).astype(np.uint8))
                    if char_img.mode != 'RGB':
                        char_img = char_img.convert('RGB')
                    
                    composite.paste(char_img, (x_pos, 0))
                    
                    # Añadir texto debajo
                    text_label = f"{char_info['predicted_letter']}"
                    conf_text = f"{char_info['confidence']*100:.0f}%"
                    
                    # Color según confianza
                    text_color = (0, 150, 0) if char_info['confidence']*100 >= confidence_threshold else (200, 0, 0)
                    
                    draw.text((x_pos + 5, char_size + 5), text_label, fill=text_color, font=font)
                    draw.text((x_pos + 5, char_size + 25), conf_text, fill=(100, 100, 100), font=font)
                    draw.text((x_pos + 5, char_size + 40), f"#{i+1}", fill=(150, 150, 150), font=font)
                
                st.image(composite, caption=f"Primeras {num_to_show} letras segmentadas (28x28 píxeles normalizadas)", use_container_width=True)
                
                # Mostrar detalles individuales en columnas
                st.markdown("---")
                st.markdown("**Detalles de cada letra:**")
                
                cols = st.columns(5)
                for i in range(num_to_show):
                    char_info = result['chars'][i]
                    with cols[i % 5]:
                        # Convertir array a imagen para mostrar
                        char_img_data = char_info['image']
                        if isinstance(char_img_data, np.ndarray):
                            # Mostrar imagen normalizada
                            img_to_show = (char_img_data * 255).astype(np.uint8)
                            st.image(Image.fromarray(img_to_show), width=80)
                        else:
                            st.image(char_img_data, width=80)
                        
                        confidence_pct = char_info['confidence'] * 100
                        if confidence_pct >= confidence_threshold:
                            st.markdown(f"✅ **Letra {i+1}**: `{char_info['predicted_letter']}`")
                        else:
                            st.markdown(f"⚠️ **Letra {i+1}**: `{char_info['predicted_letter']}`")
                        
                        st.caption(f"Confianza: {confidence_pct:.1f}%")
                        
                        # Mostrar estadísticas de la imagen
                        if isinstance(char_img_data, np.ndarray):
                            st.caption(f"Min: {char_img_data.min():.2f}")
                            st.caption(f"Max: {char_img_data.max():.2f}")
                            st.caption(f"Media: {char_img_data.mean():.2f}")
                    
                    # Nueva fila cada 5
                    if (i + 1) % 5 == 0 and i + 1 < num_to_show:
                        cols = st.columns(5)
                
                st.markdown("---")
                st.success(f"""
                **Verificación completada**: Se muestran las primeras {num_to_show} letras tal como el modelo las recibe.
                
                **Qué revisar:**
                - ✅ Las letras deben ser blancas sobre fondo negro (normalizadas para EMNIST)
                - ✅ Cada letra debe estar centrada y completa
                - ✅ No debe haber fragmentación (letras cortadas)
                - ✅ El espaciado debe separar correctamente cada carácter
                """)
            else:
                st.warning("No se detectaron caracteres en la imagen")

# TAB 2: Crear texto de prueba
with tab2:
    st.header("🖼️ Generador de Texto de Prueba")
    st.info("Crea una imagen con texto para probar el reconocimiento")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Configuración")
        
        test_text = st.text_area(
            "Escribe el texto a generar",
            value="HOLA MUNDO",
            height=100,
            max_chars=100,
            help="Solo letras mayúsculas A-Z (el modelo está entrenado con EMNIST)"
        )
        
        font_size = st.slider("Tamaño de fuente", 20, 60, 40)
        
        letter_spacing = st.slider("Espaciado entre letras", 5, 30, 15)
        
        if st.button("🎨 Generar Imagen", type="primary"):
            # Crear imagen con texto
            # Filtrar solo letras válidas
            valid_text = ''.join([c.upper() for c in test_text if c.upper().isalpha() or c == ' ' or c == '\n'])
            
            # Calcular dimensiones
            lines = valid_text.split('\n')
            max_chars_per_line = max(len(line.replace(' ', '')) for line in lines) if lines else 1
            
            img_width = max_chars_per_line * (font_size + letter_spacing) + 40
            img_height = len(lines) * (font_size + 20) + 40
            
            # Crear imagen blanca
            img = Image.new('L', (img_width, img_height), color=255)
            draw = ImageDraw.Draw(img)
            
            # Intentar usar una fuente monoespaciada
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                font = ImageFont.load_default()
            
            # Dibujar texto
            y_pos = 20
            for line in lines:
                x_pos = 20
                for char in line:
                    if char.isalpha():
                        draw.text((x_pos, y_pos), char, fill=0, font=font)
                        x_pos += font_size + letter_spacing
                    elif char == ' ':
                        x_pos += font_size + letter_spacing
                y_pos += font_size + 20
            
            st.session_state.generated_image = img
            st.success("✅ Imagen generada!")
            st.rerun()
    
    with col2:
        if 'generated_image' in st.session_state:
            st.subheader("Imagen Generada")
            st.image(st.session_state.generated_image, caption="Texto generado", use_container_width=True)
            
            if st.button("🔍 Reconocer esta imagen", type="secondary"):
                with st.spinner("Reconociendo..."):
                    text, chars, binary_img = recognize_text_from_image(
                        st.session_state.generated_image,
                        model
                    )
                    
                    st.session_state.text_result = {
                        'text': text,
                        'chars': chars,
                        'binary_img': binary_img,
                        'original_img': st.session_state.generated_image,
                        'num_chars': len(chars)
                    }
                    
                    st.success(f"✅ Reconocido: {text}")
                    st.rerun()

# TAB 3: Historial
with tab3:
    st.header("📜 Historial de Reconocimientos")
    
    if len(st.session_state.recognition_history) == 0:
        st.info("No hay reconocimientos en el historial. Procesa algunas imágenes primero.")
    else:
        st.write(f"Total de reconocimientos: **{len(st.session_state.recognition_history)}**")
        
        for i, entry in enumerate(reversed(st.session_state.recognition_history), 1):
            with st.expander(f"Reconocimiento #{len(st.session_state.recognition_history) - i + 1}"):
                st.text_area("Texto", entry['text'], height=100, key=f"hist_{i}")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Caracteres", entry['stats']['total_chars'])
                col2.metric("Líneas", entry['stats']['total_lines'])
                col3.metric("Palabras", entry['stats']['total_words'])

st.divider()
st.caption("""
💡 **Funcionalidad avanzada**: 
Este sistema implementa segmentación automática de caracteres usando proyecciones horizontales y verticales,
seguido de reconocimiento individual con el modelo CNN entrenado. Ideal para digitalizar texto manuscrito o impreso.
""")
