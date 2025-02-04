import React, { useState, useRef } from 'react';
import { Loader2, Camera, Video, RefreshCw, AlertTriangle, Download } from 'lucide-react';

function FacesDetection() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [fileURL, setFileURL] = useState('');
  const [fileType, setFileType] = useState('image');
  const [detections, setDetections] = useState([]);
  const [processedMedia, setProcessedMedia] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState(null);
  const [detectedFrames, setDetectedFrames] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentFrame, setCurrentFrame] = useState(null);
  const [downloadUrl, setDownloadUrl] = useState(null);
  const [isCompleted, setIsCompleted] = useState(false);
  const [processingTime, setProcessingTime] = useState(null);
  const [frameInterval, setFrameInterval] = useState(5);

  const fileInputRef = useRef(null);
  const videoRef = useRef(null);
  const startTimeRef = useRef(null);

  const resetDetection = () => {
    setSelectedFile(null);
    setFileURL('');
    setFileType('image');
    setDetections([]);
    setDetectedFrames([]);
    setProcessedMedia(null);
    setIsLoading(false);
    setProgress(0);
    setError(null);
    setIsProcessing(false);
    setCurrentFrame(null);
    setDownloadUrl(null);
    setIsCompleted(false);
    setProcessingTime(null);
    setFrameInterval(5);

    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const validateFileFormat = (file) => {
    const allowedImageFormats = ['image/jpeg', 'image/png', 'image/gif', 'image/bmp', 'image/webp', 'image/tiff'];
    const allowedVideoFormats = ['video/mp4', 'video/quicktime', 'video/x-msvideo', 'video/x-ms-wmv', 'video/x-flv', 'video/webm', 'video/mpeg', 'video/3gpp', 'video/x-matroska'];
    const allowedFormats = fileType === 'image' ? allowedImageFormats : allowedVideoFormats;
    
    if (!allowedFormats.includes(file.type)) {
      return `Formato no soportado. Formatos permitidos: ${fileType === 'image' ? 'JPG, PNG, GIF, BMP, WebP, TIFF' : 'MP4, MOV, AVI, WMV, FLV, WebM, MPEG, 3GP, MKV'}`;
    }
    return null;
  };

  const handleVideoResponse = async (data) => {
    try {
      if (data.progress !== undefined) {
        setProgress(data.progress);
        console.log(`Progreso actualizado: ${data.progress}%`);
      }
      
      if (data.frame) {
        setCurrentFrame(`data:image/jpeg;base64,${data.frame}`);
      }
      
      if (data.detected_frames) {
        setDetectedFrames(data.detected_frames);
      }

      if (data.download_url) {
        const fullUrl = `http://localhost:8005${data.download_url}`;
        setDownloadUrl(fullUrl);
        console.log('URL de descarga actualizada:', fullUrl);
      }

      if (data.is_completed) {
        console.log('Procesamiento completado');
        setIsProcessing(false);
        setIsCompleted(true);
        const endTime = Date.now();
        setProcessingTime(((endTime - startTimeRef.current) / 1000).toFixed(2));
        
        if (data.download_url) {
          const fullUrl = `http://localhost:8005${data.download_url}`;
          setDownloadUrl(fullUrl);
          console.log('URL de descarga final:', fullUrl);
        }
        
        setProgress(100);
      }
    } catch (error) {
      console.error('Error procesando respuesta:', error);
    }
  };

  const handleDetectFaces = async () => {
    if (!selectedFile && !fileURL) {
      setError("Por favor, selecciona un archivo o ingresa una URL.");
      return;
    }

    setIsLoading(true);
    setProgress(0);
    setDetections([]);
    setDetectedFrames([]);
    setProcessedMedia(null);
    setError(null);
    setIsProcessing(true);
    setDownloadUrl(null);
    setIsCompleted(false);
    setProcessingTime(null);
    startTimeRef.current = Date.now();

    const formData = new FormData();
    if (selectedFile) {
      formData.append('file', selectedFile);
      console.log('Archivo seleccionado:', selectedFile.name);
    } else {
      formData.append('url', fileURL);
      console.log('URL proporcionada:', fileURL);
    }

    if (fileType === 'video') {
      formData.append('frame_interval', frameInterval.toString());
      console.log('Intervalo de frames:', frameInterval);
    }

    try {
      const endpoint = fileType === 'image' ? '/api/faces/detect' : '/api/faces/detect-video';
      const response = await fetch(`http://localhost:8005${endpoint}`, {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || 'Error al procesar el archivo');
      }

      if (fileType === 'image') {
        const blob = await response.blob();
        setProcessedMedia(URL.createObjectURL(blob));
        
        const detectionsHeader = response.headers.get('X-Detections');
        if (detectionsHeader) {
          setDetections(JSON.parse(detectionsHeader));
        }

        const download_url = response.headers.get('X-Download-Url');
        if (download_url) {
          const fullUrl = `http://localhost:8005${download_url}`;
          setDownloadUrl(fullUrl);
          console.log('URL de descarga imagen:', fullUrl);
        }

        const endTime = Date.now();
        setProcessingTime(((endTime - startTimeRef.current) / 1000).toFixed(2));
        setIsCompleted(true);
        setIsProcessing(false);
        setProgress(100);
      } else {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value);
          const events = chunk.split('\n\n');

          for (const event of events) {
            if (event.startsWith('data: ')) {
              try {
                const data = JSON.parse(event.replace('data: ', ''));
                await handleVideoResponse(data);
              } catch (parseError) {
                console.error('Error parseando datos:', parseError);
              }
            }
          }
        }
      }
    } catch (err) {
      console.error("Error:", err);
      setError(err.message || "Error inesperado al procesar el archivo");
      setIsProcessing(false);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="space-y-6 p-6 bg-gray-50">
      <h2 className="text-3xl font-bold text-gray-800 mb-6">
        Detección de Rostros
      </h2>
      
      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative" role="alert">
          <AlertTriangle className="inline-block mr-2" />
          {error}
        </div>
      )}

      {isCompleted && (
        <div className="bg-green-100 border border-green-400 text-green-700 px-4 py-3 rounded relative">
          {fileType === 'video' ? 'Video exitosamente procesado' : 'Imagen exitosamente procesada'}
        </div>
      )}
      
      <div className="bg-white p-6 rounded-lg shadow-md">
        <div className="flex items-center justify-between mb-4">
          <div className="flex space-x-4">
            <button
              onClick={() => {
                setFileType('image');
                setError(null);
              }}
              className={`flex items-center space-x-2 px-4 py-2 rounded ${fileType === 'image' ? 'bg-blue-500 text-white' : 'bg-gray-200'}`}
            >
              <Camera size={20} />
              <span>Foto</span>
            </button>
            <button
              onClick={() => {
                setFileType('video');
                setError(null);
              }}
              className={`flex items-center space-x-2 px-4 py-2 rounded ${fileType === 'video' ? 'bg-blue-500 text-white' : 'bg-gray-200'}`}
            >
              <Video size={20} />
              <span>Video</span>
            </button>
          </div>
          
          <button
            onClick={resetDetection}
            className="p-2 bg-orange-200 rounded-full hover:bg-orange-300 transition"
            title="Reiniciar"
          >
            <RefreshCw size={20} className="text-orange-700" />
          </button>
        </div>

        {fileType === 'video' && (
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Intervalo de frames a procesar (1-30):
            </label>
            <input 
              type="number"
              min="1"
              max="30"
              value={frameInterval}
              onChange={(e) => setFrameInterval(Math.min(30, Math.max(1, parseInt(e.target.value) || 1)))}
              className="w-full p-2 border rounded"
            />
            <p className="mt-1 text-sm text-gray-500">
              Procesar 1 de cada {frameInterval} frames. Valores más altos = procesamiento más rápido
            </p>
          </div>
        )}

        <input
          ref={fileInputRef}
          type="file"
          onChange={(e) => {
            const file = e.target.files[0];
            if (file) {
              const formatError = validateFileFormat(file);
              if (formatError) {
                setError(formatError);
                return;
              }
              setSelectedFile(file);
              setFileURL('');
              setError(null);
            }
          }}
          className="w-full mb-4 p-2 border rounded"
          accept={fileType === 'image' ? 'image/*' : 'video/*'}
        />

        <input
          type="text"
          className="w-full p-2 border rounded mb-4"
          value={fileURL}
          onChange={(e) => {
            setFileURL(e.target.value);
            setSelectedFile(null);
            setError(null);
          }}
          placeholder="O ingresa una URL..."
        />

        <button
          onClick={handleDetectFaces}
          disabled={isLoading || (!selectedFile && !fileURL)}
          className="w-full bg-blue-600 text-white py-3 rounded hover:bg-blue-700 transition flex items-center justify-center disabled:bg-gray-400"
        >
          {isLoading ? (
            <><Loader2 className="mr-2 animate-spin" /> Procesando...</>
          ) : (
            'Detectar Rostros'
          )}
        </button>
      </div>

      {(isLoading || isProcessing) && (
        <div className="bg-white p-6 rounded-lg shadow-md">
          <div className="mb-4">
            <div className="bg-gray-200 h-3 rounded-full overflow-hidden">
              <div 
                className="bg-blue-600 h-full rounded-full transition-all duration-300"
                style={{ width: `${progress}%` }}
              />
            </div>
            <div className="flex justify-between mt-2 text-sm text-gray-600">
              <span>Progreso: {progress.toFixed(1)}%</span>
              {processingTime && (
                <span>Tiempo: {processingTime} s</span>
              )}
            </div>
          </div>
        </div>
      )}

      {isProcessing && fileType === 'video' && currentFrame && (
        <div className="mt-6 bg-white p-6 rounded-lg shadow-md">
          <h3 className="text-xl font-semibold mb-4">Procesamiento en tiempo real</h3>
          <img 
            src={currentFrame}
            alt="Frame actual" 
            className="w-full rounded-lg"
          />
        </div>
      )}

      {detectedFrames.length > 0 && (
        <div className="mt-6 bg-white p-6 rounded-lg shadow-md">
          <h3 className="text-xl font-semibold mb-4">Frames con mejor detección:</h3>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            {detectedFrames.map((frame) => (
              <div key={frame.frame_number} className="relative border rounded-lg overflow-hidden shadow-md hover:shadow-lg transition-shadow">
                <img
                  src={`data:image/jpeg;base64,${frame.thumbnail}`}
                  alt={`Frame ${frame.frame_number}`}
                  className="w-full h-auto"
                />
                <div className="absolute bottom-0 left-0 right-0 bg-black bg-opacity-70 text-white p-2">
                  <div className="text-sm font-medium">Frame {frame.frame_number}</div>
                  <div className="text-xs space-y-1">
                    <div className="font-semibold">
                      Confianza: {(frame.avg_confidence * 100).toFixed(1)}%
                    </div>
                    <div>
                      Rostros detectados: {frame.detections.length}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {((processedMedia && isCompleted) || downloadUrl) && (
        <div className="mt-6 bg-white p-6 rounded-lg shadow-md">
          <h3 className="text-xl font-semibold mb-4">Resultado final</h3>
          {fileType === 'image' && processedMedia && (
            <img 
              src={processedMedia}
              alt="Imagen procesada" 
              className="w-full rounded-lg mb-4"
            />
          )}
          {downloadUrl && (
            <div className="mt-4">
            <a 
              href={downloadUrl}
              download
              className="inline-flex items-center px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 transition"
            >
              <Download className="mr-2" />
              Descargar {fileType === 'image' ? 'Imagen' : 'Video'}
            </a>
          </div>
        )}
        {processingTime && (
          <div className="mt-2 text-sm text-gray-600">
            Tiempo de procesamiento total: {processingTime} segundos
          </div>
        )}
      </div>
    )}
  </div>
);
}

export default FacesDetection;