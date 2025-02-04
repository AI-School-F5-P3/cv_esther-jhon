// src/pages/RealtimeDetection.js

import React, { useState, useRef, useEffect } from "react";
import Webcam from "react-webcam";
import { Camera, AlertTriangle, Video, RefreshCw } from "lucide-react";

function RealtimeDetection() {
  const [mode, setMode] = useState(null);
  const [modelSize, setModelSize] = useState(null);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [error, setError] = useState(null);
  const [detections, setDetections] = useState([]);
  
  // References a elementos y a la animación
  const webcamRef = useRef(null);
  const wsRef = useRef(null);
  const canvasRef = useRef(null);
  const animationRef = useRef(null);

  const videoConstraints = {
    width: 640,
    height: 480,
    facingMode: "user"
  };

  // ------------------ 1. Conectar WebSocket ------------------
  const connectWebSocket = () => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.close();
    }

    let wsUrl;
    if (mode === "faces") {
      wsUrl = `ws://localhost:8005/ws/realtime/faces`;
    } else if (mode === "brands" && modelSize) {
      // Ejemplo: ws://localhost:8005/ws/realtime/brands?model_size=v8n
      wsUrl = `ws://localhost:8005/ws/realtime/brands?model_size=${modelSize}`;
    } else {
      return;
    }

    console.log("Conectando a WebSocket:", wsUrl);
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log("WebSocket abierto");
      // Comenzar a capturar frames en bucle
      startFrameCapture();
    };

    ws.onmessage = (evt) => {
      try {
        const data = JSON.parse(evt.data);
        if (data.error) {
          setError(data.error);
          return;
        }
        setDetections(data.detections || []);
      } catch (err) {
        console.error("Error parseando mensaje WS:", err);
      }
    };

    ws.onerror = (err) => {
      console.error("WebSocket error:", err);
      setError("Error en la conexión WebSocket");
    };

    ws.onclose = () => {
      console.log("WebSocket cerrado");
    };
  };

  // ------------------ 2. Capturar frames en bucle ------------------
  const startFrameCapture = () => {
    if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
    }

    const capture = () => {
        if (webcamRef.current && wsRef.current?.readyState === WebSocket.OPEN) {
            const imageSrc = webcamRef.current.getScreenshot();
            if (imageSrc) {
                const base64Data = imageSrc.split(",")[1];
                wsRef.current.send(base64Data);
            }
        }
        // Limitar la frecuencia de envío a 10 fps
        setTimeout(() => {
            animationRef.current = requestAnimationFrame(capture);
        }, 100); // 100 ms = 10 fps
    };
    animationRef.current = requestAnimationFrame(capture);
};

  // Conectar WS cada vez que se active la cámara
  useEffect(() => {
    if (isCameraActive && ((mode === "faces") || (mode === "brands" && modelSize))) {
      connectWebSocket();
    }
    return () => {
      // Limpiar
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
        animationRef.current = null;
      }
    };
  }, [isCameraActive, mode, modelSize]);

  // ------------------ 3. Dibujo de bounding boxes ------------------
  const drawDetections = () => {
    const canvas = canvasRef.current;
    const video = webcamRef.current?.video;
    if (!canvas || !video) return;

    const ctx = canvas.getContext("2d");
    // Ajustar tamaño
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Pintar la cámara en vivo
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Por si difiere, calcúlale un factor de escala
    const scaleX = canvas.width / video.videoWidth;
    const scaleY = canvas.height / video.videoHeight;

    // Recorrer detecciones
    detections.forEach((det) => {
        const { x, y, width, height, authorized, confidence, name, screen_time } = det;

        // Coordenadas reescaladas
        const cx = x * scaleX;
        const cy = y * scaleY;
        const w = width * scaleX;
        const h = height * scaleY;

        const x1 = cx - w / 2;
        const y1 = cy - h / 2;

        // Color
        ctx.strokeStyle = authorized ? "green" : "red";
        ctx.lineWidth = 2;
        ctx.strokeRect(x1, y1, w, h);

        const label = authorized
            ? `${name} (${(confidence * 100).toFixed(1)}%) ${screen_time}`
            : `No autorizado (${(confidence * 100).toFixed(1)}%) ${screen_time}`;

        ctx.font = "16px Arial";
        const textW = ctx.measureText(label).width;
        ctx.fillStyle = authorized
            ? "rgba(0,255,0,0.5)"
            : "rgba(255,0,0,0.5)";
        ctx.fillRect(x1, y1 - 20, textW + 8, 20);

        ctx.fillStyle = "#fff";
        ctx.fillText(label, x1 + 4, y1 - 5);
    });
};

  useEffect(() => {
    drawDetections();
  }, [detections]);

  // ------------------ 4. Handlers de la UI ------------------
  const startCamera = () => {
    if (!mode || (mode === "brands" && !modelSize)) {
      setError("Por favor, selecciona un modo de detección");
      return;
    }
    setIsCameraActive(true);
    setError(null);
  };

  const stopCamera = () => {
    setIsCameraActive(false);
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
      animationRef.current = null;
    }
    if (wsRef.current) {
      wsRef.current.close();
    }
    setDetections([]);
  };

  const selectMode = (selectedMode) => {
    setMode(selectedMode);
    if (selectedMode === "faces") {
      setModelSize(null);
    }
  };

  const reset = () => {
    stopCamera();
    setMode(null);
    setModelSize(null);
    setError(null);
    setDetections([]);
  };

  // ------------------ 5. Render ------------------
  return (
    <div className="max-w-4xl mx-auto p-6 bg-gray-50">
      <h2 className="text-3xl font-bold text-gray-800 mb-6">Detección en Tiempo Real</h2>

      {error && (
        <div className="mb-4 bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
          <AlertTriangle className="inline-block mr-2" />
          {error}
        </div>
      )}

      <div className="bg-white p-6 rounded-lg shadow-md space-y-6">
        <div className="flex justify-between items-center">
          <div className="flex gap-4">
            {/* Botón Marcas */}
            <div className="flex flex-col gap-2">
              <button
                onClick={() => selectMode("brands")}
                disabled={isCameraActive}
                className={`flex items-center gap-2 px-4 py-2 rounded ${
                  mode === "brands" ? "bg-blue-500 text-white" : "bg-gray-200"
                }`}
              >
                <Camera />
                Marcas
              </button>

              {mode === "brands" && (
                <div className="flex gap-2">
                  <button
                    onClick={() => setModelSize("v8n")}
                    disabled={isCameraActive}
                    className={`px-3 py-1 rounded text-sm ${
                      modelSize === "v8n" ? "bg-green-500 text-white" : "bg-gray-200"
                    }`}
                  >
                    YOLOv8n
                  </button>
                  <button
                    onClick={() => setModelSize("v8s")}
                    disabled={isCameraActive}
                    className={`px-3 py-1 rounded text-sm ${
                      modelSize === "v8s" ? "bg-green-500 text-white" : "bg-gray-200"
                    }`}
                  >
                    YOLOv8s
                  </button>
                </div>
              )}
            </div>

            {/* Botón Rostros */}
            <button
              onClick={() => selectMode("faces")}
              disabled={isCameraActive}
              className={`flex items-center gap-2 px-4 py-2 rounded ${
                mode === "faces" ? "bg-blue-500 text-white" : "bg-gray-200"
              }`}
            >
              <Video />
              Rostros
            </button>
          </div>

          {/* Botón Reset */}
          <button
            onClick={reset}
            className="p-2 bg-orange-200 rounded-full hover:bg-orange-300"
            title="Reiniciar"
          >
            <RefreshCw className="text-orange-700" />
          </button>
        </div>

        <div className="flex items-center justify-between">
          <div className="flex gap-4">
            {mode === "brands" && modelSize && (
              <span className="text-gray-700">
                Modelo activo: 
                <span className="ml-2 font-bold text-blue-600">
                  {modelSize === "v8n" ? "YOLOv8n" : "YOLOv8s"}
                </span>
              </span>
            )}
            {mode === "faces" && (
              <span className="text-gray-700">
                Modo: 
                <span className="ml-2 font-bold text-blue-600">
                  Detección de rostros
                </span>
              </span>
            )}
          </div>

          <button
            onClick={isCameraActive ? stopCamera : startCamera}
            disabled={!mode || (mode === "brands" && !modelSize)}
            className={`flex items-center gap-2 px-4 py-2 rounded text-white ${
              isCameraActive ? "bg-red-500" : "bg-green-500"
            } hover:opacity-90 disabled:opacity-50`}
          >
            <Camera />
            {isCameraActive ? "Detener" : "Iniciar"} Detección
          </button>
        </div>

        {/* Contenedor Webcam + Canvas */}
        <div className="relative w-[640px] h-[480px] mx-auto bg-gray-100 rounded-lg overflow-hidden">
          {isCameraActive && (
            <>
              <Webcam
                ref={webcamRef}
                audio={false}
                screenshotFormat="image/jpeg"
                videoConstraints={videoConstraints}
                className="absolute inset-0 w-full h-full object-cover"
              />
              <canvas
                ref={canvasRef}
                className="absolute inset-0 w-full h-full pointer-events-none"
              />
            </>
          )}
        </div>

        {/* Lista de detecciones */}
        {detections.length > 0 && (
          <div className="space-y-4">
            <h3 className="text-lg font-medium">Detecciones:</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {detections.map((det, idx) => {
                const isFace = (mode === "faces");
                const isAuthorized = (isFace && det.authorized);
                return (
                  <div
                    key={idx}
                    className={`p-4 rounded-lg border ${
                      isFace
                        ? (isAuthorized
                          ? "border-green-500 bg-green-50"
                          : "border-red-500 bg-red-50")
                        : "border-blue-500 bg-blue-50"
                    }`}
                  >
                    {isFace ? (
                      <>
                        <p className="font-medium">
                          {det.authorized
                            ? `Nombre: ${det.name}`
                            : "No autorizado"}
                        </p>
                        <p>Confianza: {(det.confidence * 100).toFixed(2)}%</p>
                        <p>Tiempo: {det.screen_time}</p>
                      </>
                    ) : (
                      <>
                        <p className="font-medium">
                          Marca: {det.label || "Desconocida"}
                        </p>
                        <p>Confianza: {(det.confidence * 100).toFixed(2)}%</p>
                        {det.screen_time && (
                          <p>Tiempo en pantalla: {det.screen_time}</p>
                        )}
                      </>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default RealtimeDetection;
