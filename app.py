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
from inference_sdk import InferenceHTTPClient
import face_recognition
import shutil

# Configuraciones iniciales
os.environ["STREAMLIT_SERVER_PORT"] = os.getenv("STREAMLIT_SERVER_PORT", "8501")
load_dotenv()
db_url = os.getenv("DATA_URL", "")

# Inicialización del cliente de Roboflow
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="2GQYhlGQWADhgmIt7mJA"
)

# Crear directorio para almacenar imágenes registradas si no existe
REGISTERED_FACES_DIR = "registered_faces"
if not os.path.exists(REGISTERED_FACES_DIR):
    os.makedirs(REGISTERED_FACES_DIR)

# Conexión a la base de datos
conn = psycopg2.connect(db_url)
cur = conn.cursor()

# Creación de tablas (manteniendo la estructura original y añadiendo la nueva tabla)
# Creación de tablas
cur.execute("""
    DO $$ 
    BEGIN
        -- Tabla existente para detecciones de bebidas
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
        
        -- Tabla para detecciones de caras
        CREATE TABLE IF NOT EXISTS registered_faces (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            image_path TEXT NOT NULL,
            face_encoding bytea NOT NULL,
            registration_date TIMESTAMP DEFAULT NOW(),
            is_active BOOLEAN DEFAULT TRUE
        );
        
        -- Tabla para registros de detecciones faciales
        CREATE TABLE IF NOT EXISTS face_detections (
            id SERIAL PRIMARY KEY,
            face_id INTEGER REFERENCES registered_faces(id) NULL,
            confidence FLOAT,
            detection_time TIMESTAMP DEFAULT NOW(),
            detection_type VARCHAR(50),
            duration_seconds FLOAT NULL
        );
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

# Funciones básicas de detección
def process_basic_face_detection(image):
    """Detecta rostros sin reconocimiento, solo para modo Imagen y Video"""
    try:
        result = CLIENT.infer(image, model_id="face-detection-mik1i/18")
        if not result or "predictions" not in result:
            return []
            
        # Guardar las detecciones en la base de datos
        for prediction in result["predictions"]:
            cur.execute("""
                INSERT INTO face_detections (confidence, detection_type)
                VALUES (%s, %s)
            """, (prediction["confidence"], "basic_detection"))
            conn.commit()
            
        return result["predictions"]
    except Exception as e:
        st.error(f"Error en la detección facial: {str(e)}")
        return []

def process_face_recognition(image):
    """Detecta y reconoce rostros para modo Tiempo Real"""
    try:
        # Detección inicial
        detections = process_basic_face_detection(image)
        if not detections:
            return []

        # Convertir imagen para face_recognition
        image_np = np.array(image)
        
        # Preparar ubicaciones de rostros detectados
        face_locations = []
        for det in detections:
            top = int(det["y"] - (det["height"] / 2))
            right = int(det["x"] + (det["width"] / 2))
            bottom = int(det["y"] + (det["height"] / 2))
            left = int(det["x"] - (det["width"] / 2))
            face_locations.append((top, right, bottom, left))

        # Obtener encodings de los rostros detectados
        face_encodings = face_recognition.face_encodings(image_np, face_locations)

        # Obtener rostros registrados de la base de datos
        cur.execute("SELECT id, name, face_encoding FROM registered_faces WHERE is_active = TRUE")
        registered_faces = cur.fetchall()

        # Comparar cada rostro detectado
        for idx, face_encoding in enumerate(face_encodings):
            detections[idx]["is_registered"] = False
            detections[idx]["name"] = "No registrado"
            detections[idx]["face_id"] = None
            
            for reg_id, reg_name, stored_encoding in registered_faces:
                stored_encoding_array = np.frombuffer(stored_encoding)
                if face_recognition.compare_faces([stored_encoding_array], face_encoding, tolerance=0.6)[0]:
                    detections[idx]["is_registered"] = True
                    detections[idx]["name"] = reg_name
                    detections[idx]["face_id"] = reg_id
                    break

        return detections

    except Exception as e:
        st.error(f"Error en el reconocimiento facial: {str(e)}")
        return []

def draw_detected_faces(image, detections, recognition_mode=False):
    """Dibuja los boundingboxes en la imagen"""
    image_np = np.array(image)

    for det in detections:
        x = int(det["x"])
        y = int(det["y"])
        w = int(det["width"])
        h = int(det["height"])
        conf = det["confidence"]

        # Calcular coordenadas
        x1 = int(x - w/2)
        y1 = int(y - h/2)
        x2 = int(x + w/2)
        y2 = int(y + h/2)

        if recognition_mode:
            # Modo tiempo real - verde para registrados, rojo para no registrados
            color = (0, 255, 0) if det.get("is_registered", False) else (0, 0, 255)
            label = f"{det.get('name', 'No registrado')} ({conf:.2f})"
            if det.get("is_registered", False):
                label += f" ID: {det.get('face_id', 'N/A')}"
        else:
            # Modo básico - solo detección
            color = (255, 0, 0)  # Rojo para todas las detecciones
            label = f"Rostro ({conf:.2f})"

        # Dibujar bounding box
        cv2.rectangle(image_np, (x1, y1), (x2, y2), color, 2)
        
        # Agregar etiqueta
        cv2.putText(image_np, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image_np

def register_face_from_image(name, image):
    """Registra un rostro desde una imagen"""
    try:
        # Detectar rostro
        detections = process_basic_face_detection(image)
        if not detections:
            return False, "No se detectó ningún rostro en la imagen"

        # Obtener encoding
        image_np = np.array(image)
        face_location = (
            int(detections[0]["y"] - (detections[0]["height"] / 2)),
            int(detections[0]["x"] + (detections[0]["width"] / 2)),
            int(detections[0]["y"] + (detections[0]["height"] / 2)),
            int(detections[0]["x"] - (detections[0]["width"] / 2))
        )
        face_encodings = face_recognition.face_encodings(image_np, [face_location])

        if not face_encodings:
            return False, "No se pudo obtener el encoding facial"

        # Guardar imagen
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_filename = f"{name}_{timestamp}.jpg"
        image_path = os.path.join(REGISTERED_FACES_DIR, image_filename)
        image.save(image_path)

        # Guardar en base de datos
        face_encoding_bytes = face_encodings[0].tobytes()
        cur.execute(
            "INSERT INTO registered_faces (name, image_path, face_encoding) VALUES (%s, %s, %s) RETURNING id",
            (name, image_path, face_encoding_bytes)
        )
        face_id = cur.fetchone()[0]
        conn.commit()

        return True, face_id

    except Exception as e:
        return False, f"Error al registrar el rostro: {str(e)}"

def process_video(source, is_url=False):
    """Procesa el video para detección facial"""
    try:
        if is_url:
            cap = cv2.VideoCapture(source)
        else:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(source.read())
            cap = cv2.VideoCapture(tfile.name)

        if not cap.isOpened():
            st.error("No se pudo abrir el video")
            return

        start_time = time.time()
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Variables para seguimiento de tiempo
        faces_detected = {}  # {face_id: tiempo_total}
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            current_time = (frame_count / fps) if fps > 0 else (time.time() - start_time)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            
            detections = process_basic_face_detection(image)
            if detections:
                for det in detections:
                    face_id = det.get("face_id")
                    if face_id not in faces_detected:
                        faces_detected[face_id] = 0
                    faces_detected[face_id] += 1/fps if fps > 0 else 0.033  # ~30fps si no hay fps

        # Guardar tiempo total de aparición de cada rostro
        for face_id, total_time in faces_detected.items():
            cur.execute("""
                INSERT INTO face_detections (face_id, detection_type, duration_seconds)
                VALUES (%s, %s, %s)
            """, (face_id, "video", total_time))
            conn.commit()

        cap.release()

    except Exception as e:
        st.error(f"Error al procesar el video: {str(e)}")

def register_face_interface():
    """Interfaz para registro de rostros con bounding box"""
    st.subheader("Registro de Personas")
    
    register_method = st.radio(
        "Método de registro",
        ["Subir Foto", "Usar Webcam"],
        horizontal=True
    )

    name = st.text_input("Nombre de la persona")

    if register_method == "Subir Foto":
        uploaded_file = st.file_uploader("Seleccionar foto", type=["jpg", "jpeg", "png"])
        if uploaded_file and name:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Foto subida", use_column_width=True)
            
            if st.button("Registrar", key="register_photo"):
                success, result = register_face_from_image(name, image)
                if success:
                    st.success(f"Rostro registrado exitosamente con ID: {result}")
                else:
                    st.error(result)

    else:  # Usar Webcam
        st.write("Captura desde la webcam")
        if name:
            camera_container = st.empty()
            preview_container = st.empty()  # Contenedor para la previsualización
            
            if "capture_active" not in st.session_state:
                st.session_state.capture_active = False
            if "captured_frame" not in st.session_state:
                st.session_state.captured_frame = None

            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Iniciar/Detener Captura", key="toggle_webcam_capture"):
                    st.session_state.capture_active = not st.session_state.capture_active
                    st.session_state.captured_frame = None

            with col2:
                capture_button = st.button("Capturar", key="webcam_capture_button")

            with col3:
                register_button = st.button("Registrar", key="register_captured")

            if st.session_state.capture_active:
                cap = cv2.VideoCapture(0)
                try:
                    while st.session_state.capture_active:
                        ret, frame = cap.read()
                        if ret:
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            image = Image.fromarray(frame_rgb)
                            
                            # Detectar rostros en tiempo real
                            detections = process_basic_face_detection(image)
                            if detections:
                                frame_with_detections = draw_detected_faces(image, detections)
                                camera_container.image(frame_with_detections, channels="RGB", use_column_width=True)
                                
                                # Si se presiona capturar, guardar el frame con las detecciones
                                if capture_button:
                                    st.session_state.captured_frame = frame_with_detections
                                    st.session_state.capture_active = False
                            else:
                                camera_container.image(frame_rgb, channels="RGB", use_column_width=True)
                finally:
                    cap.release()

            # Mostrar la imagen capturada si existe
            if st.session_state.captured_frame is not None:
                preview_container.image(
                    st.session_state.captured_frame, 
                    caption="Imagen capturada", 
                    use_column_width=True
                )

                # Registrar la imagen capturada
                if register_button:
                    if isinstance(st.session_state.captured_frame, np.ndarray):
                        image_pil = Image.fromarray(st.session_state.captured_frame)
                    else:
                        image_pil = st.session_state.captured_frame

                    success, result = register_face_from_image(name, image_pil)
                    if success:
                        st.success(f"Rostro registrado exitosamente con ID: {result}")
                        st.session_state.captured_frame = None  # Limpiar la imagen capturada
                    else:
                        st.error(result)

def realtime_detection():
    """Realiza detección facial en tiempo real con reconocimiento"""
    if "realtime_active" not in st.session_state:
        st.session_state.realtime_active = False

    camera_container = st.empty()
    
    if st.button("Iniciar/Detener Detección", key="toggle_realtime"):
        st.session_state.realtime_active = not st.session_state.realtime_active

    if st.session_state.realtime_active:
        cap = cv2.VideoCapture(0)
        try:
            while st.session_state.realtime_active:
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(frame_rgb)
                    
                    try:
                        detections = process_face_recognition(image)
                        if detections:
                            frame_with_detections = draw_detected_faces(image, detections, recognition_mode=True)
                            camera_container.image(frame_with_detections, channels="RGB")
                            
                            for det in detections:
                                if det.get("is_registered", False):
                                    try:
                                        cur.execute("""
                                            INSERT INTO face_detections (face_id, confidence, detection_type)
                                            VALUES (%s, %s, %s)
                                        """, (det["face_id"], det["confidence"], "realtime"))
                                        conn.commit()
                                    except Exception as db_error:
                                        st.error(f"Error al guardar detección: {str(db_error)}")
                                        continue
                        else:
                            camera_container.image(frame_rgb, channels="RGB")
                    except Exception as detection_error:
                        st.error(f"Error en la detección: {str(detection_error)}")
                        camera_container.image(frame_rgb, channels="RGB")
                else:
                    st.error("Error al acceder a la cámara")
                    break
        finally:
            cap.release()
            camera_container.empty()
            st.session_state.realtime_active = False

def main():
    st.set_page_config(
        page_title="Sistema de Detección YOLOv8",
        layout="centered"
    )

    st.sidebar.title("Configuración")
    detection_type = st.sidebar.radio(
        "Tipo de Detección",
        ["Detección de Bebidas", "Detección Facial"],
        index=0
    )

    if detection_type == "Detección de Bebidas":
        st.title("Detección de Bebidas con YOLOv8")
        
        # Configuración del modelo YOLOv8
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

    else:  # Detección Facial
        st.title("Sistema de Detección Facial")
        st.sidebar.title("Modos de Detección")
        face_mode = st.sidebar.selectbox(
            "Selecciona modo",
            ["Subir Imagen", "URL Imagen", "Subir Video", "URL Video", "Tiempo Real", "Registrar Persona"]
        )

        if face_mode == "Registrar Persona":
            register_face_interface()
        elif face_mode == "Tiempo Real":
            st.subheader("Detección Facial en Tiempo Real")
            realtime_detection()
        elif face_mode == "Subir Imagen":
            process_image_detection()
        elif face_mode == "URL Imagen":
            process_image_url()
        elif face_mode == "Subir Video":
            process_video_upload()
        else:  # URL Video
            process_video_url()

def process_image_detection():
    """Procesa imágenes subidas para detección facial"""
    st.subheader("Detección Facial en Imágenes")
    uploaded_file = st.file_uploader("Seleccionar imagen", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Imagen original", use_column_width=True)
            
            detections = process_basic_face_detection(image)
            if detections:
                image_with_detections = draw_detected_faces(image, detections)
                st.image(image_with_detections, caption="Detecciones", use_column_width=True)
                
                # Preparar imagen para descarga
                buf = io.BytesIO()
                Image.fromarray(image_with_detections).save(buf, format="JPEG")
                st.download_button(
                    label="Descargar Imagen con Detecciones",
                    data=buf.getvalue(),
                    file_name="detected_faces.jpg",
                    mime="image/jpeg"
                )
                
                st.write(f"Se encontraron {len(detections)} rostros")
                for idx, det in enumerate(detections, 1):
                    st.write(f"Rostro {idx}: Confianza {det['confidence']:.2f}")
            else:
                st.warning("No se detectaron rostros en la imagen")
                
        except Exception as e:
            st.error(f"Error al procesar la imagen: {str(e)}")

def process_image_url():
    """Procesa imágenes desde URL para detección facial"""
    st.subheader("Detección Facial desde URL")
    url = st.text_input("Ingresa la URL de la imagen")
    
    if url and st.button("Procesar"):
        try:
            response = requests.get(url)
            image = Image.open(io.BytesIO(response.content)).convert("RGB")
            st.image(image, caption="Imagen original", use_column_width=True)
            
            detections = process_basic_face_detection(image)
            if detections:
                image_with_detections = draw_detected_faces(image, detections)
                st.image(image_with_detections, caption="Detecciones", use_column_width=True)
                
                # Preparar imagen para descarga
                buf = io.BytesIO()
                Image.fromarray(image_with_detections).save(buf, format="JPEG")
                st.download_button(
                    label="Descargar Imagen con Detecciones",
                    data=buf.getvalue(),
                    file_name="detected_faces.jpg",
                    mime="image/jpeg"
                )
                
                st.write(f"Se encontraron {len(detections)} rostros")
                for idx, det in enumerate(detections, 1):
                    st.write(f"Rostro {idx}: Confianza {det['confidence']:.2f}")
            else:
                st.warning("No se detectaron rostros en la imagen")
                
        except Exception as e:
            st.error(f"Error al procesar la imagen: {str(e)}")

def process_video_upload():
    """Procesa videos subidos para detección facial"""
    st.subheader("Detección Facial en Videos")
    uploaded_file = st.file_uploader("Seleccionar video", type=["mp4", "avi", "mov"])
    
    if uploaded_file:
        try:
            st.video(uploaded_file)
            if st.button("Procesar Video"):
                with st.spinner("Procesando video..."):
                    process_video(uploaded_file, is_url=False)
                st.success("Procesamiento completado")
        except Exception as e:
            st.error(f"Error al procesar el video: {str(e)}")

def process_video_url():
    """Procesa videos desde URL para detección facial"""
    st.subheader("Detección Facial desde URL de Video")
    url = st.text_input("Ingresa la URL del video (YouTube u otros servicios)")
    
    if url and st.button("Procesar"):
        try:
            with st.spinner("Obteniendo video..."):
                video_url = get_video_url(url)
                if video_url:
                    st.video(url)
                    with st.spinner("Procesando video..."):
                        process_video(video_url, is_url=True)
                    st.success("Procesamiento completado")
                else:
                    st.error("No se pudo obtener la URL del video")
        except Exception as e:
            st.error(f"Error al procesar el video: {str(e)}")        
    
if __name__ == "__main__":
    main()