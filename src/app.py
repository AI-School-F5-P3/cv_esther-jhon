import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
import pandas as pd
import os

def load_model():
    """Carga el modelo YOLO entrenado."""
    BASE_MODEL_DIR = "/content/drive/MyDrive/Proyecto Computer Vision/Entrenamiento yolo"
    model_option = st.sidebar.radio(
        "Selecciona el modelo:",
        ("25 épocas", "50 épocas con early stopping")
    )
    
    if model_option == "25 épocas":
        model_path = os.path.join(BASE_MODEL_DIR, "25_epochs", "best_model_25epochs_640px.pt")
    else:
        model_path = os.path.join(BASE_MODEL_DIR, "50_epochs", "best_model_50epochs_640px_earlystop.pt")
    
    if not os.path.exists(model_path):
        st.error(f"Error: No se encontró el modelo en {model_path}")
        return None
    
    model = YOLO(model_path)
    return model, model_option

def process_image(image, model, conf_threshold):
    """Procesa una imagen y detecta bebidas."""
    results = model.predict(image, conf=conf_threshold)
    return results[0]

def draw_boxes(image, results):
    """Dibuja las bounding boxes y etiquetas en la imagen."""
    annotated_image = image.copy()
    detections = []

    if len(results.boxes) > 0:
        for box in results.boxes:
            # Extraemos coordenadas y confianza
            x1, y1, x2, y2 = map(int, box.xyxy[0][:4])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = results.names[class_id]

            # Guardamos la detección para la tabla
            detections.append({
                'Marca': class_name,
                'Confianza': f"{confidence:.2%}"
            })

            # Color según el nivel de confianza (más verde = más confianza)
            color = (0, int(255 * confidence), 0)

            # Dibujamos el rectángulo
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)

            # Añadimos la etiqueta
            label = f"{class_name}: {confidence:.1%}"
            cv2.putText(annotated_image, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return annotated_image, detections

def main():
    st.set_page_config(page_title="Detector de Marcas de Bebidas", layout="wide")

    st.title("🔍 Detector de Marcas de Bebidas")
    st.write("Identifica bebidas y sus marcas en imágenes o videos.")

    # Cargamos el modelo
    with st.spinner('Cargando modelo entrenado...'):
        model, model_option = load_model()
        if model is None:
            st.stop()

    st.sidebar.info(f"Modelo seleccionado: {model_option}")

    # Creamos dos columnas
    col1, col2 = st.columns([1, 2])

    with col1:
        # Selector de tipo de archivo
        file_type = st.radio("Tipo de archivo:", ["Imagen", "Video"])

        # Ajuste del umbral de confianza
        confidence_threshold = st.slider(
            "Umbral de confianza",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.05
        )

    with col2:
        if file_type == "Imagen":
            uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

            if uploaded_file is not None:
                # Procesamos la imagen
                image = Image.open(uploaded_file)
                image_np = np.array(image)
                results = process_image(image_np, model, confidence_threshold)

                # Dibujamos las detecciones
                output_image, detections = draw_boxes(image_np, results)

                # Mostramos la imagen procesada
                st.image(output_image, caption="Imagen procesada", use_column_width=True)

                # Mostramos la tabla de detecciones
                if detections:
                    st.write("### Detecciones encontradas:")
                    df = pd.DataFrame(detections)
                    st.table(df)
                else:
                    st.warning("No se detectaron marcas de bebidas en esta imagen.")

        else:  # Video
            uploaded_file = st.file_uploader("Sube un video", type=["mp4", "avi"])

            if uploaded_file is not None:
                # Guardamos temporalmente el video
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(uploaded_file.read())

                # Procesamos el video
                video = cv2.VideoCapture(tfile.name)
                stframe = st.empty()
                metrics_placeholder = st.empty()

                while video.isOpened():
                    ret, frame = video.read()
                    if not ret:
                        break

                    # Procesamos el frame
                    results = process_image(frame, model, confidence_threshold)
                    output_frame, detections = draw_boxes(frame, results)

                    # Mostramos el frame procesado
                    stframe.image(output_frame, channels="BGR", use_column_width=True)

                    # Actualizamos las métricas en tiempo real
                    if detections:
                        metrics_placeholder.table(pd.DataFrame(detections))

                video.release()

if __name__ == "__main__":
    main()



'''
#Primera versión del código
import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
import pandas as pd


def load_model():
    """Carga el modelo YOLO entrenado."""
    model_path = "models/best_model.pt"  # Cambia esta ruta según tu estructura local
    model = YOLO(model_path)  # Carga el modelo entrenado
    return model


def process_image(image, model, conf_threshold):
    """Procesa una imagen y detecta bebidas."""
    results = model.predict(image, conf=conf_threshold)
    return results[0]


def draw_boxes(image, results):
    """Dibuja las bounding boxes y etiquetas en la imagen."""
    annotated_image = image.copy()
    detections = []

    if len(results.boxes) > 0:
        for box in results.boxes:
            # Extraemos coordenadas y confianza
            x1, y1, x2, y2 = map(int, box.xyxy[0][:4])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = results.names[class_id]

            # Guardamos la detección para la tabla
            detections.append({
                'Marca': class_name,
                'Confianza': f"{confidence:.2%}"
            })

            # Color según el nivel de confianza (más verde = más confianza)
            color = (0, int(255 * confidence), 0)

            # Dibujamos el rectángulo
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)

            # Añadimos la etiqueta
            label = f"{class_name}: {confidence:.1%}"
            cv2.putText(annotated_image, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return annotated_image, detections


def main():
    st.set_page_config(page_title="Detector de Marcas de Bebidas", layout="wide")

    st.title("🔍 Detector de Marcas de Bebidas")
    st.write("Identifica bebidas y sus marcas en imágenes o videos.")

    # Cargamos el modelo
    with st.spinner('Cargando modelo entrenado...'):
        model = load_model()

    # Creamos dos columnas
    col1, col2 = st.columns([1, 2])

    with col1:
        # Selector de tipo de archivo
        file_type = st.radio("Tipo de archivo:", ["Imagen", "Video"])

        # Ajuste del umbral de confianza
        confidence_threshold = st.slider(
            "Umbral de confianza",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.05
        )

    with col2:
        if file_type == "Imagen":
            uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

            if uploaded_file is not None:
                # Procesamos la imagen
                image = Image.open(uploaded_file)
                image_np = np.array(image)
                results = process_image(image_np, model, confidence_threshold)

                # Dibujamos las detecciones
                output_image, detections = draw_boxes(image_np, results)

                # Mostramos la imagen procesada
                st.image(output_image, caption="Imagen procesada", use_column_width=True)

                # Mostramos la tabla de detecciones
                if detections:
                    st.write("### Detecciones encontradas:")
                    df = pd.DataFrame(detections)
                    st.table(df)
                else:
                    st.warning("No se detectaron marcas de bebidas en esta imagen.")

        else:  # Video
            uploaded_file = st.file_uploader("Sube un video", type=["mp4", "avi"])

            if uploaded_file is not None:
                # Guardamos temporalmente el video
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(uploaded_file.read())

                # Procesamos el video
                video = cv2.VideoCapture(tfile.name)
                stframe = st.empty()
                metrics_placeholder = st.empty()

                while video.isOpened():
                    ret, frame = video.read()
                    if not ret:
                        break

                    # Procesamos el frame
                    results = process_image(frame, model, confidence_threshold)
                    output_frame, detections = draw_boxes(frame, results)

                    # Mostramos el frame procesado
                    stframe.image(output_frame, channels="BGR", use_column_width=True)

                    # Actualizamos las métricas en tiempo real
                    if detections:
                        metrics_placeholder.table(pd.DataFrame(detections))

                video.release()


if __name__ == "__main__":
    main()

'''