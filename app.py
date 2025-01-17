import os
import streamlit as st
from dotenv import load_dotenv
import psycopg2
from ultralytics import YOLO
from PIL import Image
import cv2
import tempfile
import requests
import numpy as np
import io
from datetime import datetime
import time
import yt_dlp  # Nuevo import para manejar URLs de YouTube

# Configuraciones iniciales
os.environ["STREAMLIT_SERVER_PORT"] = os.getenv("STREAMLIT_SERVER_PORT", "8501")
load_dotenv()
db_url = os.getenv("DATA_URL", "")
conn = psycopg2.connect(db_url)
cur = conn.cursor()

# Tabla con tiempo de detección
cur.execute("""
    DO $$ 
    BEGIN
        CREATE TABLE IF NOT EXISTS detections (
            id SERIAL PRIMARY KEY,
            input_type VARCHAR(50),
            source_type VARCHAR(50),
            source_name TEXT,
            class_name VARCHAR(100),
            confidence FLOAT,
            detection_time TIMESTAMP DEFAULT NOW(),
            duration_seconds FLOAT NULL
        );
        
        IF NOT EXISTS (
            SELECT 1 
            FROM information_schema.columns 
            WHERE table_name='detections' AND column_name='duration_seconds'
        ) THEN
            ALTER TABLE detections ADD COLUMN duration_seconds FLOAT NULL;
        END IF;
    END
    $$;
""")
conn.commit()

model = None

def insert_detection(input_type, source_type, source_name, class_name, confidence, duration=None):
    # Redondear confidence a 2 decimales y duration_seconds a 4 decimales
    rounded_confidence = round(confidence, 2)
    rounded_duration = round(duration, 4) if duration is not None else None
    
    if input_type == "image":
        cur.execute("""
            INSERT INTO detections 
            (input_type, source_type, source_name, class_name, confidence)
            VALUES (%s, %s, %s, %s, %s)
        """, (input_type, source_type, source_name, class_name, rounded_confidence))
    else:
        cur.execute("""
            INSERT INTO detections 
            (input_type, source_type, source_name, class_name, confidence, duration_seconds)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (input_type, source_type, source_name, class_name, rounded_confidence, rounded_duration))
    conn.commit()

def get_video_url(url):
    """
    Obtiene la URL directa del video desde varias fuentes (YouTube, etc.)
    """
    try:
        ydl_opts = {
            'format': 'best',
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if 'url' in info:
                return info['url']
            elif 'formats' in info and len(info['formats']) > 0:
                return info['formats'][-1]['url']
    except Exception as e:
        st.error(f"Error al procesar la URL del video: {str(e)}")
        return None

def detect_on_image(img, input_type, source_type, source_name):
    global model
    if model is None:
        st.error("No se ha cargado ningún modelo YOLO.")
        return np.array(img)

    results = model.predict(img, conf=0.5, verbose=False)
    annotated_img = results[0].plot()

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            class_name = model.names[cls_id]
            insert_detection(input_type, source_type, source_name, class_name, conf)

    return annotated_img

def detect_on_video(video_path, input_type, source_type, source_name):
    global model
    if model is None:
        st.error("No se ha cargado ningún modelo YOLO.")
        return

    video_container = st.empty()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("No se pudo abrir el video. Formato no soportado o archivo corrupto.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    annotated_video_path = "annotated_video.mp4"
    out = cv2.VideoWriter(annotated_video_path, fourcc, fps if fps>0 else 30, (w, h))

    detection_times = {}
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    progress = st.progress(0, "Procesando video...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            progress.progress(frame_count / total_frames)

            results = model.predict(frame, conf=0.5, verbose=False)
            annotated_frame = results[0].plot()

            current_detections = set()
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0].item())
                    class_name = model.names[cls_id]
                    current_detections.add(class_name)
                    
                    if class_name not in detection_times:
                        detection_times[class_name] = {'start': frame_count, 'total': 0}
            
            for class_name in list(detection_times.keys()):
                if class_name in current_detections:
                    detection_times[class_name]['total'] = (frame_count - detection_times[class_name]['start']) / fps
                elif 'end' not in detection_times[class_name]:
                    duration = detection_times[class_name]['total']
                    conf = float(box.conf[0].item())
                    insert_detection(input_type, source_type, source_name, class_name, conf, duration)
                    detection_times[class_name]['end'] = True

            video_container.image(annotated_frame, channels="RGB", use_column_width=True)
            annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
            out.write(annotated_frame_bgr)

    except Exception as e:
        st.error(f"Error durante el procesamiento del video: {str(e)}")
        
    finally:
        cap.release()
        out.release()
        progress.empty()

    if os.path.exists(annotated_video_path):
        st.success("Video procesado correctamente.")
        with open(annotated_video_path, "rb") as f:
            st.download_button(
                label="Descargar Video Procesado",
                data=f,
                file_name="video_procesado.mp4",
                mime="video/mp4"
            )

def detect_realtime():
    global model
    if model is None:
        st.error("No se ha cargado ningún modelo YOLO.")
        return

    if "realtime_running" not in st.session_state:
        st.session_state.realtime_running = False
        st.session_state.detection_times = {}

    camera_container = st.empty()
    
    col1, col2 = st.columns(2)
    with col1:
        start_button = st.button("Iniciar Detección" if not st.session_state.realtime_running else "Detener Detección")
    with col2:
        st.info("Presiona 'ESC' para detener")
    
    if start_button:
        st.session_state.realtime_running = not st.session_state.realtime_running
        if st.session_state.realtime_running:
            st.rerun()
    
    if st.session_state.realtime_running:
        cap = cv2.VideoCapture(0)
        
        while st.session_state.realtime_running:
            ret, frame = cap.read()
            if ret:
                results = model.predict(frame, conf=0.5, verbose=False)
                annotated_frame = results[0].plot()

                current_detections = set()
                current_time = time.time()

                for r in results:
                    for box in r.boxes:
                        cls_id = int(box.cls[0].item())
                        conf = float(box.conf[0].item())
                        class_name = model.names[cls_id]
                        current_detections.add(class_name)
                        
                        if class_name not in st.session_state.detection_times:
                            st.session_state.detection_times[class_name] = {
                                'start': current_time,
                                'total': 0
                            }
                
                for class_name in list(st.session_state.detection_times.keys()):
                    if class_name in current_detections:
                        st.session_state.detection_times[class_name]['total'] = \
                            current_time - st.session_state.detection_times[class_name]['start']
                    elif 'end' not in st.session_state.detection_times[class_name]:
                        duration = st.session_state.detection_times[class_name]['total']
                        insert_detection("realtime", "camera", "webcam", class_name, conf, duration)
                        st.session_state.detection_times[class_name]['end'] = True

                camera_container.image(annotated_frame, channels="RGB", use_column_width=True)
                
                if cv2.waitKey(1) & 0xFF == 27:
                    st.session_state.realtime_running = False
                    break
            else:
                st.error("No se puede acceder a la cámara.")
                break
                
        cap.release()
        camera_container.empty()
        st.session_state.realtime_running = False

def main():
    st.set_page_config(
        page_title="Detección YOLOv8 - Prototipo",
        layout="centered"
    )
    st.title("Detección de Bebidas con YOLOv8")

    st.sidebar.title("Configuración")
    model_option = st.sidebar.radio(
        "Selecciona el modelo",
        ["Modelo v8n", "Modelo v8s"],
        index=0,
        horizontal=True
    )

    model_mapping = {
        "Modelo v8n": "best_yolov8n.pt",
        "Modelo v8s": "best_yolov8s.pt"
    }

    global model
    model = YOLO(model_mapping[model_option])

    st.sidebar.title("Modos de Detección")
    mode = st.sidebar.selectbox(
        "Selecciona modo",
        ["Imagen", "Video", "Tiempo Real"]
    )

    if mode == "Imagen":
        st.subheader("Detección en Imágenes")
        sub_mode = st.radio(
            "¿Cómo deseas proporcionar la imagen?",
            ("Subir archivo", "URL"),
            horizontal=True
        )
        
        if sub_mode == "Subir archivo":
            uploaded_file = st.file_uploader(
                "Selecciona una imagen",
                type=["jpg", "png", "jpeg"]
            )
            if uploaded_file:
                img = Image.open(uploaded_file).convert("RGB")
                st.image(img, caption="Imagen original", use_column_width=True)

                annotated_img = detect_on_image(
                    img, "image", "file", uploaded_file.name
                )
                st.image(annotated_img, caption="Detección", use_column_width=True)

                annotated_pil = Image.fromarray(annotated_img)
                buf = io.BytesIO()
                annotated_pil.save(buf, format="JPEG")
                byte_im = buf.getvalue()

                st.download_button(
                    label="Descargar Imagen Anotada",
                    data=byte_im,
                    file_name="annotated_image.jpg",
                    mime="image/jpeg"
                )
        
        else:  # URL
            url = st.text_input("Ingresa la URL de la imagen")
            if st.button("Procesar URL"):
                if url:
                    try:
                        resp = requests.get(url, stream=True)
                        resp.raise_for_status()
                        img = Image.open(resp.raw).convert("RGB")
                        st.image(img, caption="Imagen original", use_column_width=True)

                        annotated_img = detect_on_image(
                            img, "image", "url", url
                        )
                        st.image(annotated_img, caption="Detección", use_column_width=True)

                        annotated_pil = Image.fromarray(annotated_img)
                        buf = io.BytesIO()
                        annotated_pil.save(buf, format="JPEG")
                        byte_im = buf.getvalue()

                        st.download_button(
                            label="Descargar Imagen Anotada",
                            data=byte_im,
                            file_name="annotated_image.jpg",
                            mime="image/jpeg"
                        )
                    except Exception as e:
                        st.error(f"Error al cargar la imagen: {e}")

    elif mode == "Video":
        st.subheader("Detección en Videos")
        sub_mode = st.radio(
            "¿Cómo deseas proporcionar el video?",
            ("Subir archivo", "URL"),
            horizontal=True
        )
        
        if sub_mode == "Subir archivo":
            video_file = st.file_uploader(
                "Selecciona un video",
                type=["mp4", "avi", "mov", "mkv"]
            )
            if video_file:
                with st.spinner("Procesando video..."):
                    tfile = tempfile.NamedTemporaryFile(delete=False)
                    tfile.write(video_file.read())
                    detect_on_video(tfile.name, "video", "file", video_file.name)

        else:  # URL
            url = st.text_input("Ingresa la URL del video (Soporta YouTube y otros servicios)")
            if st.button("Procesar Video"):
                if url:
                    try:
                        with st.spinner("Obteniendo video..."):
                            video_url = get_video_url(url)
                            if video_url:
                                with st.spinner("Procesando video..."):
                                    resp = requests.get(video_url, stream=True)
                                    resp.raise_for_status()
                                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                                    for chunk in resp.iter_content(chunk_size=8192):
                                        if chunk:
                                            tfile.write(chunk)
                                    tfile.flush()
                                    detect_on_video(tfile.name, "video", "url", url)
                            else:
                                st.error("No se pudo obtener la URL del video")
                    except Exception as e:
                        st.error(f"Error al procesar el video: {str(e)}")

    else:  # Tiempo Real
        detect_realtime()

if __name__ == "__main__":
    main()