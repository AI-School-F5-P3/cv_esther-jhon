import base64
import io
import os
import uuid
import time
import json
from dotenv import load_dotenv

from fastapi import FastAPI, File, UploadFile, Form, WebSocket, WebSocketDisconnect, Response, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from typing import Optional, List
from pydantic import BaseModel

# BD
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session

# YOLO
from ultralytics import YOLO

# Roboflow (usando inference_sdk)
from inference_sdk import InferenceHTTPClient
import requests

# OpenCV, NumPy, face_recognition
import cv2
import numpy as np
import face_recognition
import uvicorn
import asyncio
import shutil

# ---------------------------------------------------------------------
# 1) Configuración general
# ---------------------------------------------------------------------
load_dotenv()

# Formatos permitidos
ALLOWED_IMAGE_FORMATS = [
    'image/jpeg', 'image/jpg', 'image/png', 'image/gif',
    'image/bmp', 'image/webp', 'image/tiff'
]

ALLOWED_VIDEO_FORMATS = [
    'video/mp4', 'video/quicktime', 'video/x-msvideo',
    'video/x-ms-wmv', 'video/x-flv', 'video/webm',
    'video/mpeg', 'video/3gpp', 'video/x-matroska'
]

app = FastAPI(title="Proyecto YOLO + React + FastAPI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3004"],  # Ajusta si tu frontend corre en otro puerto
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Clave y Modelo de Roboflow
ROBOFLOW_API_KEY = "2GQYhlGQWADhgmIt7mJA"
ROBOFLOW_MODEL_ID = "face-detection-mik1i/21"  # Ajusta al ID exacto de tu modelo en Roboflow

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=ROBOFLOW_API_KEY
)

# Configuración de la base de datos
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:6328@localhost:5432/yolo_react")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

# Directorios para almacenar archivos
REGISTERED_FACES_FOLDER = "registered_faces"
TEMP_FOLDER = "temp_files"
BRAND_FILES_FOLDER = "./brand_files"
os.makedirs(BRAND_FILES_FOLDER, exist_ok=True)
os.makedirs(REGISTERED_FACES_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)

# Modelos YOLO (para detección de Marcas)
model_paths = {
    "v8n": "best_yolov8n.pt",
    "v8s": "best_yolov8s.pt"
}
loaded_models = {
    "v8n": YOLO(model_paths["v8n"]),
    "v8s": YOLO(model_paths["v8s"])
}

# ---------------------------------------------------------------------
# 2) Esquemas
# ---------------------------------------------------------------------
class DetectResponse(BaseModel):
    label: str
    confidence: float
    x: float
    y: float
    w: float
    h: float

# ---------------------------------------------------------------------
# 3) Funciones auxiliares
# ---------------------------------------------------------------------
def get_face_encoding(image_data: bytes):
    """
    Obtiene el encoding de UN rostro (el primero) en una imagen.
    """
    np_arr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        return None

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_img)
    if face_locations:
        encodings = face_recognition.face_encodings(rgb_img, face_locations)
        if encodings:
            return encodings[0]
    return None

def check_face_authorization(face_encoding: np.ndarray):
    """
    Verifica si un rostro está autorizado comparándolo con la BD.
    """
    if face_encoding is None:
        return {"authorized": False, "name": "No autorizado"}
        
    session = SessionLocal()
    try:
        registered_faces = session.execute(
            text("SELECT name, face_encoding FROM registered_faces WHERE is_active = TRUE")
        ).fetchall()
        
        for face in registered_faces:
            if face.face_encoding:
                stored_encoding = np.frombuffer(face.face_encoding, dtype=np.float64)
                match = face_recognition.compare_faces([stored_encoding], face_encoding, tolerance=0.6)[0]
                if match:
                    return {"authorized": True, "name": face.name}
        
        return {"authorized": False, "name": "No autorizado"}
    finally:
        session.close()

def draw_yolo_boxes(img_cv2: np.ndarray, boxes, model) -> np.ndarray:
    """
    Dibuja bounding boxes devueltas por YOLO.
    """
    for box in boxes:
        x_center, y_center, w, h = box.xywh[0].tolist()
        conf = float(box.conf[0])
        label_id = int(box.cls[0])
        label_name = model.names.get(label_id, f"class_{label_id}")
        screen_time = getattr(box, 'time', 0)  # Obtiene tiempo si existe

        x1 = int(x_center - w/2)
        y1 = int(y_center - h/2)
        x2 = int(x_center + w/2)
        y2 = int(y_center + h/2)

        color = (0, 255, 0)
        cv2.rectangle(img_cv2, (x1, y1), (x2, y2), color, 2)
        
        # Texto actualizado con tiempo
        text = f"{label_name} {conf:.2f} {screen_time:.1f}s"
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(img_cv2, (x1, y1-20), (x1 + text_w, y1), color, -1)
        cv2.putText(img_cv2, text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return img_cv2

def draw_boxes_with_auth(img_cv2: np.ndarray, predictions: list) -> np.ndarray:
    """
    Dibuja bounding boxes con autorización, aplicando un scale_factor
    basado en la altura relativa del rostro.
    """
    for pred in predictions:
        x = pred.get("x", 0)
        y = pred.get("y", 0)
        w = pred.get("width", 0)
        h = pred.get("height", 0)
        name = pred.get("name", "No autorizado")
        authorized = pred.get("authorized", False)
        conf = pred.get("confidence", 0)
        screen_time = pred.get("screen_time", 0)

        # Ajustar el tamaño del bounding box según la distancia
        scale_factor = 1.0 + (h / img_cv2.shape[0])  # Ajuste según altura rostro
        w = int(w * scale_factor)
        h = int(h * scale_factor)

        x1 = int(x - w/2)
        y1 = int(y - h/2)
        x2 = int(x + w/2)
        y2 = int(y + h/2)

        # Verde si autorizado, rojo si no
        color = (0, 255, 0) if authorized else (0, 0, 255)
        
        cv2.rectangle(img_cv2, (x1, y1), (x2, y2), color, 2)
        
        # Texto con tiempo
        label_text = f"{name} {conf:.2f} {screen_time:.1f}s"
        (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(img_cv2, (x1, y1-20), (x1 + text_w, y1), color, -1)
        cv2.putText(img_cv2, label_text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return img_cv2

def draw_face_boxes_no_auth(img_cv2: np.ndarray, predictions: list) -> np.ndarray:
    """
    Dibuja bounding boxes de rostros SIN realizar autorización.
    """
    for pred in predictions:
        x = pred.get("x", 0)
        y = pred.get("y", 0)
        w = pred.get("width", 0)
        h = pred.get("height", 0)
        conf = pred.get("confidence", 0)

        x1 = int(x - w/2)
        y1 = int(y - h/2)
        x2 = int(x + w/2)
        y2 = int(y + h/2)

        color = (0, 255, 0)
        cv2.rectangle(img_cv2, (x1, y1), (x2, y2), color, 2)

        text = f"face {conf:.2f}"
        cv2.rectangle(img_cv2, (x1, y1-20), (x1 + len(text)*9, y1), color, -1)
        cv2.putText(img_cv2, text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
    return img_cv2

def process_detections(predictions: list, image_cv2: np.ndarray):
    """
    Convierte las predicciones de Roboflow en un formato usable (x,y,width,height,confidence)
    y dibuja bounding boxes sin autorización. Devuelve (imagen_con_boxes, lista_de_detecciones).
    """
    detections_list = []
    for p in predictions:
        det = {
            "x": p["x"],
            "y": p["y"],
            "width": p["width"],
            "height": p["height"],
            "confidence": p["confidence"]
        }
        detections_list.append(det)

    result_image = draw_face_boxes_no_auth(image_cv2, detections_list)
    return result_image, detections_list

def clean_temp_folder():
    """
    Limpia archivos temporales de más de 1 hora.
    """
    for folder in [TEMP_FOLDER, BRAND_FILES_FOLDER]:
        try:
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                if os.path.getmtime(file_path) < time.time() - 3600:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
        except Exception as e:
            print(f"Error limpiando {folder}: {str(e)}")

def get_safe_filename(filename: str) -> str:
    """
    Genera un nombre de archivo seguro. Si el nombre está vacío, genera uno único.
    """
    base = os.path.basename(filename)
    name, ext = os.path.splitext(base)
    if not name:
        name = str(uuid.uuid4())
    return name

# ---------------------------------------------------------------------
# 4) API Endpoints
# ---------------------------------------------------------------------
@app.get("/")
async def root():
    return {
        "mensaje": "API de Detección",
        "documentacion": "/docs",
        "health": "/api/health"
    }

@app.get("/api/health")
async def health_check():
    return {"status": "ok", "mensaje": "Servidor funcionando correctamente"}

# ---------------- (1) Detección de Marcas (YOLO) ----------------
@app.post("/api/brands/detect")
async def detect_brands_image(
   model_size: str = Form(...),
   file: UploadFile = File(None),
   url: str = Form(None)
):
   session = SessionLocal()
   start_time = time.time()
   file_folder = None

   try:
       if model_size not in loaded_models:
           return JSONResponse({"error": "Modelo inválido. Use 'v8n' o 'v8s'."}, status_code=400)

       try:
           if file:
               if file.content_type not in ALLOWED_IMAGE_FORMATS:
                   return JSONResponse({"error": "Formato de imagen no soportado"}, status_code=400)
               image_data = await file.read()
               source_name = get_safe_filename(file.filename)
           elif url:
               response = requests.get(url)
               if response.status_code != 200:
                   return JSONResponse({"error": "No se pudo obtener la URL"}, status_code=400)
               content_type = response.headers.get('content-type', '')
               if content_type not in ALLOWED_IMAGE_FORMATS:
                   return JSONResponse({"error": "Formato de URL no soportado"}, status_code=400)
               image_data = response.content
               source_name = get_safe_filename(url)
           else:
               return JSONResponse({"error": "Debe proporcionar un archivo o URL"}, status_code=400)

           npimg = np.frombuffer(image_data, np.uint8)
           img_cv2 = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
           if img_cv2 is None:
               return JSONResponse({"error": "No se pudo decodificar la imagen"}, status_code=400)

           file_folder = os.path.join(BRAND_FILES_FOLDER, source_name)
           os.makedirs(file_folder, exist_ok=True)

           original_path = os.path.join(file_folder, "original.jpg")
           processed_path = os.path.join(file_folder, "processed.jpg")
           cv2.imwrite(original_path, img_cv2)

           model = loaded_models[model_size]
           results = model.predict(img_cv2)[0]

           detections = []
           for r in results.boxes:
               x_center, y_center, w, h = r.xywh[0].tolist()
               conf = float(r.conf[0])
               label_id = int(r.cls[0])
               label_name = model.names.get(label_id, f"class_{label_id}")
               detections.append({
                   "label": label_name,
                   "confidence": conf,
                   "x": x_center,
                   "y": y_center,
                   "w": w,
                   "h": h
               })

           img_with_boxes = draw_yolo_boxes(img_cv2, results.boxes, model)
           cv2.imwrite(processed_path, img_with_boxes)

           duration = time.time() - start_time
           avg_confidence = sum(d["confidence"] for d in detections) / len(detections) if detections else 0

           session.execute(
               text("""
                   INSERT INTO detections 
                   (input_type, source_type, source_name, class_name, confidence, duration_seconds)
                   VALUES (:input_type, :source_type, :source_name, :class_name, :confidence, :duration_seconds)
               """),
               {
                   "input_type": "image",
                   "source_type": "file" if file else "url",
                   "source_name": source_name,
                   "class_name": "brand",
                   "confidence": avg_confidence,
                   "duration_seconds": duration
               }
           )
           session.commit()

           download_url = f"/api/brands/download/{source_name}/processed.jpg"

           return FileResponse(
               processed_path,
               media_type="image/jpeg",
               filename="processed.jpg",
               headers={
                   "X-Detections": json.dumps(detections),
                   "X-Download-Url": download_url,
                   "Access-Control-Expose-Headers": "X-Detections, X-Download-Url"
               }
           )

       except Exception as e:
           return JSONResponse({"error": str(e)}, status_code=400)

   finally:
       session.close()
       if file_folder:
           async def cleanup():
               await asyncio.sleep(3600)
               try:
                   if os.path.exists(file_folder):
                       shutil.rmtree(file_folder)
               except Exception as e:
                   print(f"Error limpiando archivos: {str(e)}")
           asyncio.create_task(cleanup())

@app.post("/api/brands/detect-video")
async def detect_brands_video(
   model_size: str = Form(...),
   file: UploadFile = File(None),
   url: str = Form(None),
   frame_interval: int = Form(5)
):
   session = SessionLocal()
   start_time = time.time()
   video_folder = None
   cap = None
   out = None

   try:
       if model_size not in loaded_models:
           return JSONResponse({"error": "Modelo inválido. Use 'v8n' o 'v8s'."}, status_code=400)

       frame_interval = max(1, min(frame_interval, 30))

       try:
           if file:
               if file.content_type not in ALLOWED_VIDEO_FORMATS:
                   return JSONResponse({"error": "Formato de video no soportado"}, status_code=400)
               video_data = await file.read()
               source_name = get_safe_filename(file.filename)
               print(f"\nProcesando video: {source_name}")
           elif url:
               response = requests.get(url)
               if response.status_code != 200:
                   return JSONResponse({"error": "No se pudo obtener la URL"}, status_code=400)
               content_type = response.headers.get('content-type', '')
               if content_type not in ALLOWED_VIDEO_FORMATS:
                   return JSONResponse({"error": "Formato de URL no soportado"}, status_code=400)
               video_data = response.content
               source_name = get_safe_filename(url)
               print(f"\nProcesando video desde URL: {url}")
           else:
               return JSONResponse({"error": "Debe proporcionar un archivo o URL"}, status_code=400)

           video_name = get_safe_filename(source_name)
           video_folder = os.path.join(BRAND_FILES_FOLDER, video_name)
           frames_folder = os.path.join(video_folder, "frames")
           detected_frames_folder = os.path.join(video_folder, "detected_frames")

           for folder in [video_folder, frames_folder, detected_frames_folder]:
               os.makedirs(folder, exist_ok=True)

           input_path = os.path.join(video_folder, "original.mp4")
           output_path = os.path.join(video_folder, "processed.mp4")

           with open(input_path, "wb") as f:
               f.write(video_data)

           cap = cv2.VideoCapture(input_path)
           if not cap.isOpened():
               raise Exception("No se pudo abrir el video")

           fps = int(cap.get(cv2.CAP_PROP_FPS))
           width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
           height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
           total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

           print(f"Info del video:")
           print(f"- Frames totales: {total_frames}")
           print(f"- FPS: {fps}")
           print(f"- Resolución: {width}x{height}\n")

           if width > 1280:
               scale = 1280 / width
               width = 1280
               height = int(height * scale)
               print(f"Redimensionando a: {width}x{height}")

           out = cv2.VideoWriter(
               output_path,
               cv2.VideoWriter_fourcc(*'XVID'),
               fps,
               (width, height)
           )
           if not out.isOpened():
               raise Exception("No se pudo crear el archivo de salida")

           async def process_video():
               frame_count = 0
               processed_count = 0
               total_confidence = 0
               num_detections = 0
               frames_to_process = total_frames // frame_interval
               last_reported_progress = -1

               try:
                   model = loaded_models[model_size]
                   print(f"Usando modelo: {model_size}")

                   while frame_count < total_frames:
                       ret, frame = cap.read()
                       if not ret:
                           break

                       frame_count += 1

                       if frame_count % frame_interval == 0:
                           processed_count += 1

                           if width != frame.shape[1] or height != frame.shape[0]:
                               frame = cv2.resize(frame, (width, height))

                           try:
                               results = model.predict(frame)[0]
                               detections = []

                               for r in results.boxes:
                                   x_center, y_center, w, h = r.xywh[0].tolist()
                                   conf = float(r.conf[0])
                                   label_id = int(r.cls[0])
                                   label_name = model.names.get(label_id, f"class_{label_id}")
                                   detections.append({
                                       "label": label_name,
                                       "confidence": conf,
                                       "x": x_center,
                                       "y": y_center,
                                       "w": w,
                                       "h": h
                                   })

                               processed_frame = draw_yolo_boxes(frame, results.boxes, model)

                               if detections:
                                   frame_confidence = sum(d["confidence"] for d in detections) / len(detections)
                                   total_confidence += frame_confidence
                                   num_detections += len(detections)

                                   frame_path = os.path.join(detected_frames_folder, f"frame_{frame_count:04d}.jpg")
                                   cv2.imwrite(frame_path, processed_frame)

                                   _, buffer = cv2.imencode('.jpg', processed_frame)
                                   frame_base64 = base64.b64encode(buffer).decode('utf-8')

                               out.write(processed_frame)

                               current_progress = int(min(100, (processed_count / frames_to_process) * 100))
                               if current_progress > last_reported_progress:
                                   last_reported_progress = current_progress
                                   print(f"Progreso: {current_progress}% - Frame {frame_count}/{total_frames}")
                                   data = {
                                       'progress': current_progress,
                                       'frame': frame_base64 if 'frame_base64' in locals() else None,
                                       'detections': detections if detections else []
                                   }
                                   yield f"data: {json.dumps(data)}\n\n"

                           except Exception as e:
                               print(f"Error procesando frame {frame_count}: {str(e)}")
                               out.write(frame)
                       else:
                           out.write(frame)

                   duration = time.time() - start_time
                   avg_confidence = total_confidence / num_detections if num_detections else 0

                   print("\n=== PROCESAMIENTO COMPLETADO ===")
                   print(f"Total frames procesados: {processed_count}/{total_frames}")
                   print(f"Total detecciones: {num_detections}")
                   print(f"Confianza promedio: {avg_confidence:.2f}")
                   print(f"Duración total: {duration:.2f} segundos")
                   print("===============================\n")

                   session.execute(
                       text("""
                           INSERT INTO detections 
                           (input_type, source_type, source_name, class_name, confidence, duration_seconds)
                           VALUES (:input_type, :source_type, :source_name, :class_name, :confidence, :duration_seconds)
                       """),
                       {
                           "input_type": "video",
                           "source_type": "file" if file else "url",
                           "source_name": source_name,
                           "class_name": "brand",
                           "confidence": avg_confidence,
                           "duration_seconds": duration
                       }
                   )
                   session.commit()

                   download_url = f"/api/brands/download/{video_name}/processed.mp4"

                   final_data = {
                       'is_completed': True,
                       'total_frames': total_frames,
                       'frames_processed': processed_count,
                       'total_detections': num_detections,
                       'avg_confidence': avg_confidence,
                       'download_url': download_url,
                       'progress': 100,
                       'message': 'Video procesado exitosamente'
                   }
                   
                   yield f"data: {json.dumps(final_data)}\n\n"

               except Exception as e:
                   print(f"Error en procesamiento: {str(e)}")
                   raise
               finally:
                   if cap and cap.isOpened():
                       cap.release()
                   if out and out.isOpened():
                       out.release()

           return StreamingResponse(
               process_video(),
               media_type="text/event-stream",
               headers={
                   "Cache-Control": "no-cache",
                   "X-Accel-Buffering": "no",
                   "Content-Type": "text/event-stream",
                   "Access-Control-Allow-Origin": "*",
                   "Access-Control-Expose-Headers": "X-Download-Url",
                   "Access-Control-Allow-Headers": "X-Download-Url",
                   "Access-Control-Allow-Methods": "POST, GET, OPTIONS"
               }
           )

       except Exception as e:
           print(f"ERROR: {str(e)}")
           if session:
               session.rollback()
           return JSONResponse({"error": str(e)}, status_code=400)

   finally:
       session.close()
       if video_folder:
           async def cleanup():
               await asyncio.sleep(3600)
               try:
                   if os.path.exists(video_folder):
                       shutil.rmtree(video_folder)
               except Exception as e:
                   print(f"Error limpiando archivos: {str(e)}")
           asyncio.create_task(cleanup())

# ---------------------------------------------------------------------
# 1.1) Endpoint de Descarga de Archivos Procesados de marcas
# ---------------------------------------------------------------------
@app.get("/api/brands/download/{file_name}/{processed_file}")
async def download_brands_file(file_name: str, processed_file: str):
   """Endpoint para descargar archivos procesados de brands."""
   file_path = os.path.join(BRAND_FILES_FOLDER, file_name, processed_file)
   if not os.path.exists(file_path):
       raise HTTPException(status_code=404, detail="Archivo no encontrado")

   _, ext = os.path.splitext(processed_file)
   ext = ext.lower()
   media_types = {
       '.jpg': 'image/jpeg',
       '.jpeg': 'image/jpeg',
       '.png': 'image/png',
       '.gif': 'image/gif',
       '.bmp': 'image/bmp',
       '.webp': 'image/webp',
       '.tiff': 'image/tiff',
       '.mp4': 'video/mp4',
       '.mov': 'video/quicktime',
       '.avi': 'video/x-msvideo',
       '.wmv': 'video/x-ms-wmv',
       '.flv': 'video/x-flv',
       '.webm': 'video/webm',
       '.mpeg': 'video/mpeg',
       '.3gp': 'video/3gpp',
       '.mkv': 'video/x-matroska'
   }
   media_type = media_types.get(ext, 'application/octet-stream')

   return FileResponse(path=file_path, filename=processed_file, media_type=media_type)

# --------------- (2) Rostros Offline (SIN auth en imagen/video) --------------
@app.post("/api/faces/detect")
async def detect_faces_image(
    request: Request,
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None)
):
    session = SessionLocal()
    start_time = time.time()
    file_folder = None
    
    try:
        ALLOWED_IMAGE_FORMATS = [
            'image/jpeg', 'image/jpg', 'image/png', 'image/gif',
            'image/bmp', 'image/webp', 'image/tiff'
        ]
        
        print(f"Iniciando detección - Timestamp: {start_time}")

        if not file and not url:
            return JSONResponse({"error": "No se proporcionó archivo ni URL"}, status_code=400)

        try:
            if file:
                if file.content_type not in ALLOWED_IMAGE_FORMATS:
                    return JSONResponse({
                        "error": f"Formato no soportado. Formatos permitidos: JPG, PNG, GIF, BMP, WebP, TIFF"
                    }, status_code=400)
                data = await file.read()
                source_name = get_safe_filename(file.filename)
                print(f"Archivo recibido: {source_name}")
            elif url:
                try:
                    response = requests.get(url)
                    content_type = response.headers.get('content-type', '')
                    if content_type not in ALLOWED_IMAGE_FORMATS:
                        return JSONResponse({"error": "Formato de URL no soportado"}, status_code=400)
                    data = response.content
                    source_name = get_safe_filename(url)
                    print(f"URL procesada: {url}")
                except requests.exceptions.RequestException as e:
                    return JSONResponse({"error": f"Error obteniendo URL: {str(e)}"}, status_code=400)

            image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                return JSONResponse({"error": "No se pudo procesar la imagen"}, status_code=400)

            # Crear carpeta específica
            file_name = get_safe_filename(source_name)
            file_folder = os.path.join(TEMP_FOLDER, file_name)
            os.makedirs(file_folder, exist_ok=True)

            # Guardar original y procesar
            original_path = os.path.join(file_folder, "original.jpg")
            processed_path = os.path.join(file_folder, "processed.jpg")
            cv2.imwrite(original_path, image, [cv2.IMWRITE_JPEG_QUALITY, 100])

            results = CLIENT.infer(original_path, model_id=ROBOFLOW_MODEL_ID)
            processed_image, detections = process_detections(results["predictions"], image)
            cv2.imwrite(processed_path, processed_image, [cv2.IMWRITE_JPEG_QUALITY, 100])

            # Registro en BD
            duration = time.time() - start_time
            avg_confidence = sum(d["confidence"] for d in detections) / len(detections) if detections else 0

            session.execute(
                text("""
                    INSERT INTO detections 
                    (input_type, source_type, source_name, class_name, confidence, duration_seconds)
                    VALUES (:input_type, :source_type, :source_name, :class_name, :confidence, :duration_seconds)
                """),
                {
                    "input_type": "image",
                    "source_type": "file" if file else "url",
                    "source_name": source_name,
                    "class_name": "face",
                    "confidence": avg_confidence,
                    "duration_seconds": duration
                }
            )
            session.commit()

            # Generar URL de descarga
            download_url = f"/api/faces/download/{file_name}/processed.jpg"

            # Enviar respuesta con headers expuestos
            _, buffer = cv2.imencode('.jpg', processed_image, [cv2.IMWRITE_JPEG_QUALITY, 100])
            return Response(
                content=buffer.tobytes(),
                media_type="image/jpeg",
                headers={
                    "X-Detections": json.dumps(detections),
                    "X-Download-Url": download_url,
                    "Access-Control-Expose-Headers": "X-Detections, X-Download-Url",
                    "Content-Disposition": f"inline; filename=processed.jpg"
                }
            )

        except Exception as e:
            print(f"Error procesando imagen: {str(e)}")
            return JSONResponse({"error": str(e)}, status_code=400)

    except Exception as e:
        print(f"Error general: {str(e)}")
        session.rollback()
        return JSONResponse({"error": str(e)}, status_code=400)
    finally:
        session.close()
        if file_folder:
            async def cleanup():
                await asyncio.sleep(3600)
                try:
                    if os.path.exists(file_folder):
                        shutil.rmtree(file_folder)
                except Exception as e:
                    print(f"Error limpiando carpeta: {str(e)}")
            asyncio.create_task(cleanup())

@app.post("/api/faces/detect-video")
async def detect_faces_video(
    request: Request,
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None),
    frame_interval: int = Form(5)
):
    session = SessionLocal()
    start_time = time.time()
    video_folder = None
    cap = None
    out = None
    
    try:
        frame_interval = max(1, min(frame_interval, 30))
        
        ALLOWED_VIDEO_FORMATS = [
            'video/mp4', 'video/quicktime', 'video/x-msvideo',
            'video/x-ms-wmv', 'video/x-flv', 'video/webm',
            'video/mpeg', 'video/3gpp', 'video/x-matroska'
        ]

        if file:
            if file.content_type not in ALLOWED_VIDEO_FORMATS:
                return JSONResponse({
                    "error": f"Formato no soportado. Formatos permitidos: MP4, MOV, AVI, WMV, FLV, WebM, MPEG, 3GP, MKV"
                }, status_code=400)
            video_data = await file.read()
            video_source = get_safe_filename(file.filename)
        elif url:
            try:
                response = requests.get(url)
                content_type = response.headers.get('content-type', '')
                if content_type not in ALLOWED_VIDEO_FORMATS:
                    return JSONResponse({"error": f"Formato de URL no soportado"}, status_code=400)
                video_data = response.content
                video_source = get_safe_filename(url)
            except Exception as e:
                return JSONResponse({"error": f"Error obteniendo URL: {str(e)}"}, status_code=400)
        else:
            return JSONResponse({"error": "No se proporcionó archivo ni URL"}, status_code=400)

        video_name = get_safe_filename(video_source)
        video_folder = os.path.join(TEMP_FOLDER, video_name)
        frames_folder = os.path.join(video_folder, "frames")
        detected_frames_folder = os.path.join(video_folder, "detected_frames")
        
        for folder in [video_folder, frames_folder, detected_frames_folder]:
            os.makedirs(folder, exist_ok=True)

        input_path = os.path.join(video_folder, "original.mp4")
        output_path = os.path.join(video_folder, "processed.mp4")
        
        with open(input_path, "wb") as f:
            f.write(video_data)

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return JSONResponse({"error": "No se pudo abrir el video"}, status_code=400)

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if width > 640:
            scale = 640 / width
            width = 640
            height = int(height * scale)

        if width <= 0 or height <= 0:
            return JSONResponse({"error": "Dimensiones de video inválidas"}, status_code=400)

        # Configuración mejorada del writer
        output_fps = int(cap.get(cv2.CAP_PROP_FPS))  # Usar FPS original del video
        try:
            out = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*'XVID'),
                output_fps,    # No limitar FPS
                (width, height),
                True
            )
            if not out.isOpened():
                raise Exception("No se pudo crear el archivo de salida")
        except Exception as e:
            return JSONResponse({"error": f"Error configurando video: {str(e)}"}, status_code=400)

        async def process_video():
            frame_count = 0
            processed_count = 0
            total_confidence = 0
            num_detections = 0
            best_frames = []
            last_detections = None
            frames_to_process = total_frames // frame_interval
            last_reported_progress = -1
            
            try:
                while frame_count < total_frames:
                    ret, frame = cap.read()
                    if not ret:
                        print(f"Fin del video en frame {frame_count}/{total_frames}")
                        break

                    frame_count += 1
                    current_progress = min(100, (frame_count / total_frames) * 100)

                    if frame_count % frame_interval == 0:
                        processed_count += 1

                        if width != frame.shape[1] or height != frame.shape[0]:
                            frame = cv2.resize(frame, (width, height))

                        frame_path = os.path.join(frames_folder, f"frame_{frame_count:04d}.jpg")
                        cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

                        try:
                            results = CLIENT.infer(frame_path, model_id=ROBOFLOW_MODEL_ID)
                            processed_frame, frame_detections = process_detections(
                                results["predictions"], frame
                            )
                            last_detections = results["predictions"]

                            if frame_detections:
                                frame_avg_confidence = sum(d["confidence"] for d in frame_detections) / len(frame_detections)
                                
                                detected_frame_path = os.path.join(
                                    detected_frames_folder, 
                                    f"detected_{frame_count:04d}.jpg"
                                )
                                cv2.imwrite(detected_frame_path, processed_frame)
                                
                                thumbnail = cv2.resize(processed_frame, (160, 90))
                                _, thumb_buffer = cv2.imencode('.jpg', thumbnail)
                                thumbnail_base64 = base64.b64encode(thumb_buffer).decode('utf-8')
                                
                                frame_data = {
                                    'frame_number': frame_count,
                                    'thumbnail': thumbnail_base64,
                                    'detections': frame_detections,
                                    'avg_confidence': frame_avg_confidence
                                }
                                
                                best_frames.append(frame_data)
                                best_frames = sorted(
                                    best_frames,
                                    key=lambda x: x['avg_confidence'],
                                    reverse=True
                                )[:6]

                                total_confidence += frame_avg_confidence
                                num_detections += len(frame_detections)

                            out.write(processed_frame)

                            current_progress = int(min(100, (processed_count / frames_to_process) * 100))
                            
                            # Solo enviar actualización si el progreso ha cambiado
                            if current_progress > last_reported_progress:
                                last_reported_progress = current_progress
                                
                                _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                                frame_base64 = base64.b64encode(buffer).decode('utf-8')

                                download_url = f"/api/faces/download/{video_name}/processed.mp4"

                                data = {
                                    'frame': frame_base64,
                                    'progress': current_progress,
                                    'detections': frame_detections,
                                    'detected_frames': best_frames,
                                    'frame_number': frame_count,
                                    'total_frames': total_frames,
                                    'frames_processed': processed_count,
                                    'total_frames_to_process': frames_to_process,
                                    'is_completed': False,
                                    'download_url': download_url
                                }
                                
                                yield f"data: {json.dumps(data)}\n\n"

                        except Exception as e:
                            print(f"Error procesando frame {frame_count}: {str(e)}")
                            if last_detections:
                                processed_frame, _ = process_detections(last_detections, frame)
                            else:
                                processed_frame = frame
                            out.write(processed_frame)
                            continue
                    else:
                        if last_detections:
                            processed_frame, _ = process_detections(last_detections, frame)
                        else:
                            processed_frame = frame
                        out.write(processed_frame)

                # Asegurar que el video se complete correctamente
                out.release()
                print(f"Video procesado completamente: {frame_count}/{total_frames} frames")

                duration = time.time() - start_time
                avg_confidence = total_confidence / num_detections if num_detections else 0

                # Registrar en base de datos
                session.execute(
                    text("""
                        INSERT INTO detections 
                        (input_type, source_type, source_name, class_name, confidence, duration_seconds)
                        VALUES (:input_type, :source_type, :source_name, :class_name, :confidence, :duration_seconds)
                    """),
                    {
                        "input_type": "video",
                        "source_type": "file" if file else "url",
                        "source_name": video_source,
                        "class_name": "face",
                        "confidence": avg_confidence,
                        "duration_seconds": duration
                    }
                )
                session.commit()

                # Enviar información final
                download_url = f"/api/faces/download/{video_name}/processed.mp4"
                
                final_data = {
                    'is_completed': True,
                    'detected_frames': best_frames,
                    'total_frames': total_frames,
                    'frames_processed': processed_count,
                    'total_frames_to_process': frames_to_process,
                    'total_detections': num_detections,
                    'avg_confidence': avg_confidence,
                    'download_url': download_url,
                    'progress': 100
                }
                
                print("Enviando datos finales con URL de descarga:", download_url)
                yield f"data: {json.dumps(final_data)}\n\n"

            except Exception as e:
                print(f"Error en el procesamiento del video: {str(e)}")
                raise
            finally:
                if cap and cap.isOpened():
                    cap.release()
                if out and out.isOpened():
                    out.release()

        return StreamingResponse(
            process_video(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Content-Type": "text/event-stream",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Expose-Headers": "X-Download-Url, Content-Disposition",
                "Access-Control-Allow-Headers": "X-Download-Url, Content-Disposition",
                "Access-Control-Allow-Methods": "POST, GET, OPTIONS"
            }
        )

    except Exception as e:
        print(f"Error general: {str(e)}")
        if session:
            session.rollback()
        return JSONResponse({"error": str(e)}, status_code=400)
    finally:
        session.close()
        
        if video_folder:
            async def cleanup():
                await asyncio.sleep(3600)
                try:
                    if os.path.exists(video_folder):
                        shutil.rmtree(video_folder)
                except Exception as e:
                    print(f"Error limpiando carpeta: {str(e)}")
            asyncio.create_task(cleanup())

# -------------------- (3) Registro de Rostro con encoding --------------------
@app.post("/api/faces/register")
async def register_face(
    file: UploadFile = File(...),
    person_name: str = Form(None),
    is_preview: bool = Form(False)
):
    session = SessionLocal() if not is_preview else None
    try:
        image_data = await file.read()

        # Detectar rostro con Roboflow
        with open("temp_image.jpg", "wb") as f:
            f.write(image_data)
        
        results = CLIENT.infer("temp_image.jpg", model_id=ROBOFLOW_MODEL_ID)
        os.remove("temp_image.jpg")
        
        if not results.get("predictions"):
            return JSONResponse({"error": "No se detectó ningún rostro"}, status_code=400)

        detection = max(results["predictions"], key=lambda x: x["confidence"])
        x, y, width, height = detection["x"], detection["y"], detection["width"], detection["height"]

        # Procesar imagen con bounding box
        np_arr = np.frombuffer(image_data, np.uint8)
        img_cv2 = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img_cv2 is None:
            return JSONResponse({"error": "Imagen inválida"}, status_code=400)

        # Dibujar texto y bounding box
        start_point = (int(x - width / 2), int(y - height / 2))
        end_point = (int(x + width / 2), int(y + height / 2))
        
        cv2.rectangle(img_cv2, start_point, end_point, (0, 255, 0), 2)
        
        confidence = detection["confidence"] * 100
        label = f"{person_name if person_name else 'Face'} {confidence:.1f}%"
        
        text_pos = (start_point[0], start_point[1] - 10)
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(img_cv2, 
                     (text_pos[0], text_pos[1] - text_h - 4),
                     (text_pos[0] + text_w, text_pos[1] + 4),
                     (0, 255, 0), -1)
        
        cv2.putText(img_cv2, label, text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        if is_preview:
            # Solo devolver imagen procesada para preview
            _, buffer = cv2.imencode('.jpg', img_cv2)
            return Response(content=buffer.tobytes(), media_type="image/jpeg")

        if not person_name:
            return JSONResponse({"error": "Se requiere nombre para el registro"}, status_code=400)

        face_encoding = get_face_encoding(image_data)
        if face_encoding is None:
            return JSONResponse({"error": "No se detectó rostro en la imagen"}, status_code=400)

        # Verificar registro existente
        existing_face = session.execute(
            text("SELECT id FROM registered_faces WHERE name = :name AND is_active = TRUE"),
            {"name": person_name}
        ).fetchone()

        if existing_face:
            session.execute(
                text("UPDATE registered_faces SET is_active = FALSE WHERE id = :id"),
                {"id": existing_face[0]}
            )

        user_folder = os.path.join(REGISTERED_FACES_FOLDER, person_name)
        os.makedirs(user_folder, exist_ok=True)

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        original_filename = f"original_{timestamp}.jpg"
        processed_filename = f"processed_{timestamp}.jpg"
        
        original_path = os.path.join(user_folder, original_filename)
        processed_path = os.path.join(user_folder, processed_filename)

        cv2.imwrite(original_path, cv2.imdecode(np_arr, cv2.IMREAD_COLOR))
        cv2.imwrite(processed_path, img_cv2)

        session.execute(
            text("""
                INSERT INTO registered_faces 
                (name, image_path, face_encoding, is_active, registration_date)
                VALUES (:name, :path, :encoding, TRUE, CURRENT_TIMESTAMP)
            """),
            {
                "name": person_name,
                "path": processed_path,
                "encoding": face_encoding.tobytes()
            }
        )
        session.commit()

        return {
            "status": "success",
            "message": "Rostro registrado correctamente",
            "name": person_name,
            "image_path": processed_path
        }

    except Exception as e:
        print(f"Error: {str(e)}")
        if session:
            session.rollback()
        return JSONResponse({"error": str(e)}, status_code=400)
    finally:
        if session:
            session.close()


# -------------------- (4) WebSockets con autorización (tiempo real) --------------------
@app.websocket("/ws/realtime/faces")
async def websocket_realtime_faces(websocket: WebSocket):
    await websocket.accept()
    print("Iniciando detección de rostros en tiempo real")
    session = SessionLocal()

    # Diccionario para trackear rostros
    active_faces = {}
    DISAPPEAR_THRESHOLD = 1.5  # Segundos
    next_tracker_id = 0

    try:
        while True:
            try:
                data = await websocket.receive_text()
                frame_bytes = base64.b64decode(data)
                current_time = time.time()

                # Decodificar imagen
                npimg = np.frombuffer(frame_bytes, np.uint8)
                img_cv2 = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
                if img_cv2 is None:
                    continue

                # Detectar rostros con Roboflow
                cv2.imwrite("temp_frame.jpg", img_cv2)
                result = CLIENT.infer("temp_frame.jpg", model_id=ROBOFLOW_MODEL_ID)
                os.remove("temp_frame.jpg")

                predictions = result.get("predictions", [])
                processed_faces = []

                for pred in predictions:
                    x_center = pred["x"]
                    y_center = pred["y"]
                    w = pred["width"]
                    h = pred["height"]
                    conf_roboflow = float(pred["confidence"])

                    # Recortar subimagen
                    x1 = int(x_center - w / 2)
                    y1 = int(y_center - h / 2)
                    x2 = int(x_center + w / 2)
                    y2 = int(y_center + h / 2)
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(img_cv2.shape[1], x2)
                    y2 = min(img_cv2.shape[0], y2)

                    face_region = img_cv2[y1:y2, x1:x2]
                    if face_region.size == 0:
                        continue

                    rgb_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
                    face_encs = face_recognition.face_encodings(rgb_face)
                    if face_encs:
                        face_enc = face_encs[0]
                        # Buscar si coincide con un rostro trackeado
                        tracker_found = None
                        for tracker_id, info in active_faces.items():
                            match = face_recognition.compare_faces([info["encoding"]], face_enc, tolerance=0.6)[0]
                            if match:
                                tracker_found = tracker_id
                                break

                        if tracker_found is not None:
                            info = active_faces[tracker_found]
                            # Si pasa mucho tiempo -> se reinicia
                            if (current_time - info["last_seen"]) > DISAPPEAR_THRESHOLD:
                                info["start_time"] = current_time
                                info["accum_time"] = 0.0

                            info["accum_time"] = current_time - info["start_time"]
                            info["last_seen"] = current_time

                            name = info["name"]
                            authorized = info["authorized"]
                            screen_time = info["accum_time"]
                        else:
                            # Nuevo rostro
                            auth_result = check_face_authorization(face_enc)
                            name = auth_result["name"]
                            authorized = auth_result["authorized"]

                            active_faces[next_tracker_id] = {
                                "encoding": face_enc,
                                "name": name,
                                "authorized": authorized,
                                "start_time": current_time,
                                "last_seen": current_time,
                                "accum_time": 0.0
                            }
                            tracker_found = next_tracker_id
                            next_tracker_id += 1
                            screen_time = 0.0

                        processed_faces.append({
                            "x": x_center,
                            "y": y_center,
                            "width": w,
                            "height": h,
                            "confidence": conf_roboflow,
                            "authorized": authorized,
                            "name": name,
                            "screen_time": f"{screen_time:.1f}s"
                        })
                    else:
                        # No se pudo encodar
                        processed_faces.append({
                            "x": x_center,
                            "y": y_center,
                            "width": w,
                            "height": h,
                            "confidence": conf_roboflow,
                            "authorized": False,
                            "name": "No autorizado",
                            "screen_time": "0.0s"
                        })

                # Limpiar trackers que no aparecieron en este frame
                to_remove = []
                for tracker_id, info in active_faces.items():
                    if (current_time - info["last_seen"]) > DISAPPEAR_THRESHOLD * 2:
                        to_remove.append(tracker_id)
                for r in to_remove:
                    del active_faces[r]

                # Enviar detecciones al frontend
                await websocket.send_json({"detections": processed_faces})

            except WebSocketDisconnect:
                print("Conexión cerrada por el cliente")
                break
            except Exception as e:
                print(f"Error en procesamiento: {str(e)}")
                continue
    finally:
        session.close()
        print("Conexión cerrada")


# -------------------- (4.1) WebSockets para deteccion de marcas (tiempo real) --------------------
@app.websocket("/ws/realtime/brands")
async def websocket_realtime_brands(websocket: WebSocket, model_size: str):
    await websocket.accept()
    print(f"Iniciando detección de marcas en tiempo real - Modelo: {model_size}")
    detection_history = {}
    
    try:
        if model_size not in loaded_models:
            await websocket.send_json({"error": "Modelo no válido"})
            return
            
        model = loaded_models[model_size]
        print(f"Usando modelo: {model_size}")
        
        while True:
            try:
                data = await websocket.receive_text()
                frame_bytes = base64.b64decode(data)
                current_time = time.time()
                
                npimg = np.frombuffer(frame_bytes, np.uint8)
                img_cv2 = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
                
                # Detectar marcas
                results = model.predict(img_cv2, conf=0.25)[0]  # threshold
                detections = []
                
                for box in results.boxes:
                    x_center, y_center, w, h = box.xywh[0].tolist()
                    conf = float(box.conf[0])
                    label_id = int(box.cls[0])
                    label_name = model.names.get(label_id, f"class_{label_id}")

                    detection_id = f"brand_{label_name}"
                    if detection_id not in detection_history:
                        detection_history[detection_id] = {
                            'start_time': current_time,
                            'last_seen': current_time
                        }
                    detection_history[detection_id]['last_seen'] = current_time
                    screen_time = current_time - detection_history[detection_id]['start_time']
                    
                    detections.append({
                        "label": label_name,
                        "confidence": conf,
                        "x": x_center,
                        "y": y_center,
                        "w": w,
                        "h": h,
                        "screen_time": f"{screen_time:.1f}s"
                    })
                
                if detections:
                    print(f"Marcas detectadas: {len(detections)}")
                
                # Limpiar detecciones antiguas
                current_brands = {f"brand_{det['label']}" for det in detections}
                for det_id in list(detection_history.keys()):
                    if det_id not in current_brands:
                        if current_time - detection_history[det_id]['last_seen'] > 1.0:
                            del detection_history[det_id]
                
                await websocket.send_json({"detections": detections})

            except WebSocketDisconnect:
                print("Conexión cerrada")
                break
            except Exception as e:
                print(f"Error en procesamiento: {str(e)}")
                continue
                
    finally:
        print("Conexión cerrada")

# ---------------------------------------------------------------------
# 5) Endpoint de Descarga de Archivos Procesados faces
# ---------------------------------------------------------------------
@app.get("/api/faces/download/{file_name}/{processed_file}")
async def download_file(file_name: str, processed_file: str):
    """
    Endpoint para descargar archivos procesados (imágenes o videos).
    """
    file_path = os.path.join(TEMP_FOLDER, file_name, processed_file)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Archivo no encontrado")

    _, ext = os.path.splitext(processed_file)
    ext = ext.lower()
    media_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.bmp': 'image/bmp',
        '.webp': 'image/webp',
        '.tiff': 'image/tiff',
        '.mp4': 'video/mp4',
        '.mov': 'video/quicktime',
        '.avi': 'video/x-msvideo',
        '.wmv': 'video/x-ms-wmv',
        '.flv': 'video/x-flv',
        '.webm': 'video/webm',
        '.mpeg': 'video/mpeg',
        '.3gp': 'video/3gpp',
        '.mkv': 'video/x-matroska'
    }
    media_type = media_types.get(ext, 'application/octet-stream')

    return FileResponse(path=file_path, filename=processed_file, media_type=media_type)

# ---------------------------------------------------------------------
# Ejecución principal
# ---------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8005)
