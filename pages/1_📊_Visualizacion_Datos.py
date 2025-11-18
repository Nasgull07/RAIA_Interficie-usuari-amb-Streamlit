import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

st.set_page_config(page_title="Visualización de Datos", page_icon="📊", layout="wide")

from utils.data_loader import (
    load_train_data,
    load_test_data,
    get_dataset_info,
    label_to_letter,
    get_class_distribution
)

st.title("📊 Visualización de Datos EMNIST")
st.markdown("""
Explora el dataset EMNIST Letters de forma interactiva con visualizaciones y estadísticas detalladas.
""")

# Cargar información del dataset
dataset_info = get_dataset_info()

# Sidebar para control
with st.sidebar:
    st.header("⚙️ Configuración")
    
    # Widget: Slider para número de muestras
    sample_size = st.slider(
        "Tamaño de muestra",
        min_value=100,
        max_value=10000,
        value=1000,
        step=100,
        help="Número de muestras a cargar para visualización"
    )
    
    # Widget: Select box para tipo de datos
    data_type = st.selectbox(
        "Tipo de datos",
        ["Entrenamiento", "Prueba"],
        help="Selecciona qué dataset visualizar"
    )
    
    load_button = st.button("📥 Cargar Datos", width='stretch')

# Inicializar session_state para datos
if 'viz_data' not in st.session_state:
    st.session_state.viz_data = None

# Cargar datos
if load_button:
    with st.spinner(f"Cargando {sample_size} muestras..."):
        if data_type == "Entrenamiento":
            X, y = load_train_data(sample_size=sample_size)
        else:
            X, y = load_test_data(sample_size=sample_size)
        
        if X is not None and y is not None:
            st.session_state.viz_data = (X, y, data_type)
            st.success(f"✅ Cargadas {len(X)} muestras!")
            st.rerun()

# Mostrar visualizaciones si hay datos cargados
if st.session_state.viz_data is not None:
    X, y, data_type_loaded = st.session_state.viz_data
    
    # Tabs para diferentes visualizaciones
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Distribución",
        "🖼️ Muestras",
        "📊 Estadísticas",
        "🔍 Explorador"
    ])
    
    # TAB 1: Distribución de clases
    with tab1:
        st.header("Distribución de Letras")
        
        # Calcular distribución
        distribution = get_class_distribution(y)
        
        # Crear DataFrame
        df_dist = pd.DataFrame(list(distribution.items()), columns=['Letra', 'Cantidad'])
        df_dist = df_dist.sort_values('Letra')
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Gráfico de barras interactivo
            fig = px.bar(
                df_dist,
                x='Letra',
                y='Cantidad',
                title='Frecuencia de cada letra en el dataset',
                labels={'Cantidad': 'Número de muestras'},
                color='Cantidad',
                color_continuous_scale='viridis'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Estadísticas")
            st.metric("Total muestras", len(y))
            st.metric("Clases únicas", len(distribution))
            st.metric("Promedio por clase", f"{len(y)/len(distribution):.1f}")
            st.metric("Más común", max(distribution, key=distribution.get))
            st.metric("Menos común", min(distribution, key=distribution.get))
            
            # Widget: Expander con tabla detallada
            with st.expander("Ver tabla completa"):
                st.dataframe(df_dist, hide_index=True, use_container_width=True)
    
    # TAB 2: Muestras visuales
    with tab2:
        st.header("Galería de Muestras")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # Widget: Select para filtrar por letra
            selected_letter = st.selectbox(
                "Filtrar por letra",
                ["Todas"] + list(sorted(set([label_to_letter(l) for l in y])))
            )
            
            num_samples = st.slider(
                "Número de imágenes",
                min_value=4,
                max_value=20,
                value=12,
                step=4
            )
        
        with col2:
            # Filtrar por letra si se seleccionó una
            if selected_letter != "Todas":
                indices = [i for i, label in enumerate(y) if label_to_letter(label) == selected_letter]
            else:
                indices = list(range(len(y)))
            
            # Seleccionar muestras aleatorias
            if len(indices) > 0:
                sample_indices = np.random.choice(indices, min(num_samples, len(indices)), replace=False)
                
                # Crear grid de imágenes
                cols_per_row = 4
                rows = (len(sample_indices) + cols_per_row - 1) // cols_per_row
                
                for row in range(rows):
                    cols = st.columns(cols_per_row)
                    for col_idx in range(cols_per_row):
                        idx = row * cols_per_row + col_idx
                        if idx < len(sample_indices):
                            with cols[col_idx]:
                                img_idx = sample_indices[idx]
                                letter = label_to_letter(y[img_idx])
                                st.image(X[img_idx], caption=f"{letter}", use_container_width=True)
    
    # TAB 3: Estadísticas avanzadas
    with tab3:
        st.header("Análisis Estadístico")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Intensidad de Píxeles")
            
            # Calcular estadísticas de píxeles
            pixel_means = X.reshape(len(X), -1).mean(axis=1)
            pixel_stds = X.reshape(len(X), -1).std(axis=1)
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=pixel_means, name='Media', nbinsx=50))
            fig.update_layout(
                title="Distribución de intensidad media por imagen",
                xaxis_title="Intensidad media",
                yaxis_title="Frecuencia",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Varianza de Píxeles")
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=pixel_stds, name='Desviación estándar', nbinsx=50))
            fig.update_layout(
                title="Distribución de varianza por imagen",
                xaxis_title="Desviación estándar",
                yaxis_title="Frecuencia",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Imagen promedio por letra
        st.subheader("Imagen Promedio por Letra")
        
        # Calcular imágenes promedio
        letters = sorted(set([label_to_letter(l) for l in y]))
        avg_images = []
        
        for letter in letters[:13]:  # Primera mitad
            indices = [i for i, l in enumerate(y) if label_to_letter(l) == letter]
            if indices:
                avg_img = X[indices].mean(axis=0)
                avg_images.append((letter, avg_img))
        
        if avg_images:
            cols = st.columns(len(avg_images))
            for i, (letter, avg_img) in enumerate(avg_images):
                with cols[i]:
                    st.image(avg_img, caption=letter, use_container_width=True)
    
    # TAB 4: Explorador interactivo
    with tab4:
        st.header("Explorador Interactivo")
        
        # Widget: Number input para seleccionar índice
        col1, col2 = st.columns([1, 2])
        
        with col1:
            img_index = st.number_input(
                "Índice de imagen",
                min_value=0,
                max_value=len(X)-1,
                value=0,
                help="Selecciona el índice de la imagen a visualizar"
            )
            
            if st.button("🎲 Imagen Aleatoria"):
                img_index = np.random.randint(0, len(X))
                st.rerun()
            
            # Información de la imagen
            st.subheader("Información")
            st.metric("Letra", label_to_letter(y[img_index]))
            st.metric("Etiqueta numérica", int(y[img_index]))
            st.metric("Intensidad media", f"{X[img_index].mean():.3f}")
            st.metric("Desv. estándar", f"{X[img_index].std():.3f}")
        
        with col2:
            st.subheader("Visualización")
            
            # Mostrar imagen ampliada
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
            
            # Imagen normal
            ax1.imshow(X[img_index], cmap='gray')
            ax1.set_title(f"Imagen: {label_to_letter(y[img_index])}")
            ax1.axis('off')
            
            # Histograma de píxeles
            ax2.hist(X[img_index].flatten(), bins=50, color='blue', alpha=0.7)
            ax2.set_title("Distribución de píxeles")
            ax2.set_xlabel("Intensidad")
            ax2.set_ylabel("Frecuencia")
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

else:
    # Si no hay datos cargados, mostrar instrucciones
    st.info("""
    👆 **Usa el panel lateral para comenzar:**
    
    1. Ajusta el tamaño de la muestra
    2. Selecciona el tipo de datos
    3. Haz clic en "Cargar Datos"
    
    Las visualizaciones aparecerán aquí una vez cargados los datos.
    """)
    
    # Mostrar información del dataset disponible
    st.header("📚 Información del Dataset Disponible")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Clases", dataset_info['num_classes'])
    with col2:
        st.metric("Tamaño imagen", f"{dataset_info['image_shape'][0]}x{dataset_info['image_shape'][1]}")
    with col3:
        if dataset_info.get('train_samples') != 'Desconocido':
            st.metric("Muestras totales", f"{dataset_info['train_samples']:,}")

st.divider()
st.caption("💡 **Elemento**: Visualización de Datos - Permite explorar y entender el dataset antes del entrenamiento")
