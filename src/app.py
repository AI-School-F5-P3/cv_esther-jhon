import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import tempfile

# Configuración de la página
st.set_page_config(
    page_title="Detector de Marcas de Bebidas",
    page_icon="🥤",
    layout="wide"
)

# Función para cargar el modelo seleccionado
@st.cache_resource
def load_model(epochs):
    try:
        if epochs == 25:
            model_path = "C:/Users/Administrator/Desktop/Proyecto_Proyecto CV - Detección de Objetos/cv_esther-jhon/src/models/25epochs/best_model_25epochs_640px.pt"
        else:
            model_path = "C:/Users/Administrator/Desktop/Proyecto_Proyecto CV - Detección de Objetos/cv_esther-jhon/src/models/50epochs/best_model_50epochs_640px_earlystop.pt"
        
        model = YOLO(model_path)
        return model, None
    except Exception as e:
        return None, str(e)

def correct_colors(frame):
    # Ajustar el balance de blancos
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final

def main():
    st.title("🥤 Detector de Marcas de Bebidas🥤")
    
    # Selector de modelo en la barra lateral
    st.sidebar.header("⚙️ Configuración del Modelo")
    epochs = st.sidebar.radio(
        "Seleccionar Modelo YOLOv8s:",
        [25, 50],
        format_func=lambda x: f"Modelo {x} épocas de entrenamiento"
    )
    
    # Cargar el modelo seleccionado
    model, error = load_model(epochs)
    
    if error:
        st.error(f"Error al cargar el modelo: {error}")
        st.error("Por favor, verifica que:")
        st.error("1. La ruta al modelo es correcta")
        st.error("2. El archivo del modelo existe")
        st.stop()
    else:
        st.sidebar.success(f"✅ Modelo YOLOv8s de {epochs} épocas cargado correctamente!")
    
    # Menú principal
    option = st.radio(
        "Seleccione el modo de detección:",
        ["📸 Subir Imagen", "🎥 Subir Video", "📹 Cámara en Vivo"],
        horizontal=True
    )
    
    # Configuración de confianza
    confidence = st.sidebar.slider(
        "Umbral de Confianza",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05
    )
    
    # Procesamiento según la opción seleccionada
    if option == "📸 Subir Imagen":
        uploaded_file = st.file_uploader("Selecciona una imagen...", type=['png', 'jpg', 'jpeg'])
        
        if uploaded_file is not None:
            # Mostrar imagen original
            col1, col2 = st.columns(2)
            image = Image.open(uploaded_file)
            with col1:
                st.subheader("Imagen Original")
                st.image(image, use_container_width=True)
            
            # Botón para realizar la detección
            if st.button("Realizar Detección"):
                with st.spinner("Procesando imagen..."):
                    # Realizar predicción con el umbral de confianza seleccionado
                    results = model.predict(image, conf=confidence)[0]
                    
                    # Mostrar imagen con detecciones
                    with col2:
                        st.subheader("Detecciones Encontradas")
                        st.image(results.plot(), use_container_width=True)
                    
                    # Mostrar información de las detecciones
                    st.subheader("Detalle de las predicciones:")
                    if len(results.boxes) > 0:
                        for box in results.boxes:
                            conf = float(box.conf[0])
                            cls = int(box.cls[0])
                            name = results.names[cls]
                            st.write(f"- Marca detectada: {name} (Confianza: {conf:.2%})")
                    else:
                        st.info("No se detectaron marcas de bebidas en la imagen con el umbral de confianza actual.")
    
    elif option == "🎥 Subir Video":
        video_file = st.file_uploader("Selecciona un video...", type=['mp4', 'avi', 'mov'])
        
        if video_file is not None:
            # Guardar el video subido temporalmente
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_file.read())
            
            st.video(video_file)  # Mostrar video original
            
            if st.button("Procesar Video"):
                stframe = st.empty()
                cap = cv2.VideoCapture(tfile.name)
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame = correct_colors(frame)
                    
                    # Realizar predicción con el umbral de confianza seleccionado
                    results = model.predict(frame, conf=confidence)[0]
                    
                    # Mostrar frame con detecciones
                    stframe.image(results.plot(), channels="BGR", use_container_width=True)
                
                cap.release()
    
    else:  # Cámara en Vivo
        st.write("### Detección en tiempo real")
        run = st.checkbox("Activar Cámara")
        
        if run:
            stframe = st.empty()
            cap = cv2.VideoCapture(0)  # 0 es la cámara por defecto
            
            while run:
                ret, frame = cap.read()
                if not ret:
                    st.error("Error al acceder a la cámara")
                    break
                
                frame = correct_colors(frame)
                
                # Realizar predicción con el umbral de confianza seleccionado
                results = model.predict(frame, conf=confidence)[0]
                
                # Dibujar las detecciones en el frame
                annotated_frame = results.plot()
                
                # Mostrar frame con detecciones
                stframe.image(annotated_frame, channels="BGR", use_container_width=True)
                
                # Mostrar información de las detecciones
                detections = []
                for box in results.boxes:
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    name = results.names[cls]
                    detections.append(f"{name} ({conf:.2%})")
                
                if detections:
                    st.write("Detecciones:", ", ".join(detections))
                else:
                    st.write("No se detectaron marcas de bebidas")
            
            cap.release()

if __name__ == "__main__":
    main()






