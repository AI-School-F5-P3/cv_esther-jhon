import React, { useState, useRef } from 'react';
import { Loader2, Camera, Video, RefreshCw, AlertTriangle, Download } from 'lucide-react';

function BrandsDetection() {
 const [selectedFile, setSelectedFile] = useState(null);
 const [fileURL, setFileURL] = useState('');
 const [fileType, setFileType] = useState('image');
 const [isLoading, setIsLoading] = useState(false);
 const [progress, setProgress] = useState(0);
 const [error, setError] = useState(null);
 const [currentFrame, setCurrentFrame] = useState(null);
 const [downloadUrl, setDownloadUrl] = useState(null);
 const [isCompleted, setIsCompleted] = useState(false);
 const [processingTime, setProcessingTime] = useState(null);
 const [frameInterval, setFrameInterval] = useState(5);
 const [detections, setDetections] = useState([]);
 const [modelSize, setModelSize] = useState(null);
 const [message, setMessage] = useState(null);

 const fileInputRef = useRef(null);
 const startTimeRef = useRef(null);

 const resetDetection = () => {
   setSelectedFile(null);
   setFileURL('');
   setFileType('image');
   setIsLoading(false);
   setProgress(0);
   setError(null);
   setCurrentFrame(null);
   setDownloadUrl(null);
   setIsCompleted(false);
   setProcessingTime(null);
   setFrameInterval(5);
   setDetections([]);
   setModelSize(null);
   setMessage(null);
   
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

 const handleResponse = async (data) => {
   try {
     if (data.progress !== undefined) {
       setProgress(data.progress);
     }

     if (data.frame) {
       setCurrentFrame(`data:image/jpeg;base64,${data.frame}`);
     }

     if (data.detections) {
       setDetections(data.detections);
     }

     if (data.message) {
       setMessage(data.message);
     }

     if (data.download_url) {
       const fullUrl = `http://localhost:8005${data.download_url}`;
       setDownloadUrl(fullUrl);
     }

     if (data.is_completed) {
       setIsCompleted(true);
       const endTime = Date.now();
       setProcessingTime(((endTime - startTimeRef.current) / 1000).toFixed(2));
       setProgress(100);
       setIsLoading(false);
     }
   } catch (error) {
     console.error('Error procesando respuesta:', error);
   }
 };

 const handleDetectBrands = async () => {
   if (!selectedFile && !fileURL) {
     setError("Por favor, selecciona un archivo o ingresa una URL.");
     return;
   }

   if (!modelSize) {
     setError("Por favor, selecciona un modelo YOLO.");
     return;
   }

   setIsLoading(true);
   setProgress(0);
   setError(null);
   setMessage(null);
   setIsCompleted(false);
   setProcessingTime(null);
   setDetections([]);
   startTimeRef.current = Date.now();

   const formData = new FormData();
   formData.append('model_size', modelSize);
   
   if (selectedFile) {
     formData.append('file', selectedFile);
   } else {
     formData.append('url', fileURL);
   }

   if (fileType === 'video') {
     formData.append('frame_interval', frameInterval.toString());
   }

   try {
     const endpoint = fileType === 'image' ? '/api/brands/detect' : '/api/brands/detect-video';
     const response = await fetch(`http://localhost:8005${endpoint}`, {
       method: 'POST',
       body: formData,
     });

     if (!response.ok) {
       const errorText = await response.text();
       throw new Error(errorText || 'Error al procesar el archivo');
     }

     if (fileType === 'image') {
       const blob = await response.blob();
       const mediaUrl = URL.createObjectURL(blob);
       setDownloadUrl(mediaUrl);
       
       const detectionsHeader = response.headers.get('X-Detections');
       if (detectionsHeader) {
         setDetections(JSON.parse(detectionsHeader));
       }
       
       const endTime = Date.now();
       setProcessingTime(((endTime - startTimeRef.current) / 1000).toFixed(2));
       setIsCompleted(true);
       setIsLoading(false);
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
               await handleResponse(data);
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
     setIsLoading(false);
   }
 };

 return (
   <div className="space-y-6 p-6 bg-gray-50">
     <h2 className="text-3xl font-bold text-gray-800 mb-6">
       Detección de Marcas
     </h2>

     {error && (
       <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative" role="alert">
         <AlertTriangle className="inline-block mr-2" />
         {error}
       </div>
     )}

     {message && (
       <div className="bg-blue-100 border border-blue-400 text-blue-700 px-4 py-3 rounded relative">
         {message}
       </div>
     )}

     {isCompleted && (
       <div className="bg-green-100 border border-green-400 text-green-700 px-4 py-3 rounded relative">
         {fileType === 'video' ? 'Video exitosamente procesado' : 'Imagen exitosamente procesada'}
         {detections.length > 0 && ` - ${detections.length} marcas detectadas`}
       </div>
     )}

     <div className="bg-white p-6 rounded-lg shadow-md">
       <div className="flex items-center justify-between mb-4">
         <div className="flex space-x-4">
           <button
             onClick={() => {
               setFileType('image');
               setError(null);
               setMessage(null);
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
               setMessage(null);
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

       <div className="mb-4">
         <label className="block text-sm font-medium text-gray-700 mb-1">
           Selecciona un modelo YOLO:
         </label>
         <div className="flex space-x-4">
           <button
             onClick={() => setModelSize('v8n')}
             className={`px-4 py-2 rounded transition ${
               modelSize === 'v8n' ? 'bg-blue-500 text-white' : 'bg-gray-200'
             }`}
           >
             YOLOv8n (Rápido)
           </button>
           <button
             onClick={() => setModelSize('v8s')}
             className={`px-4 py-2 rounded transition ${
               modelSize === 'v8s' ? 'bg-blue-500 text-white' : 'bg-gray-200'
             }`}
           >
             YOLOv8s (Preciso)
           </button>
         </div>
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
         onClick={handleDetectBrands}
         disabled={isLoading || (!selectedFile && !fileURL) || !modelSize}
         className="w-full bg-blue-600 text-white py-3 rounded hover:bg-blue-700 transition flex items-center justify-center disabled:bg-gray-400"
       >
         {isLoading ? (
           <><Loader2 className="mr-2 animate-spin" /> Procesando... {progress}%</>
         ) : (
           'Detectar Marcas'
         )}
       </button>
     </div>

     {downloadUrl && fileType === 'image' && (
       <div className="mt-6 bg-white p-6 rounded-lg shadow-md">
         <h3 className="text-xl font-semibold mb-4">Resultado de la detección</h3>
         <img 
           src={downloadUrl}
           alt="Imagen procesada" 
           className="w-full rounded-lg mb-4"
         />
         <div className="flex justify-between items-center">
           <a 
             href={downloadUrl}
             download
             className="inline-flex items-center px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 transition"
           >
             <Download className="mr-2" />
             Descargar Imagen
           </a>
           {processingTime && (
             <div className="text-sm text-gray-600">
               Tiempo de procesamiento: {processingTime} segundos
             </div>
           )}
         </div>
       </div>
     )}

     {(isLoading || currentFrame) && fileType === 'video' && (
       <div className="mt-6 bg-white p-6 rounded-lg shadow-md">
         <h3 className="text-xl font-semibold mb-4">
           Procesamiento en tiempo real
           {detections.length > 0 && ` - ${detections.length} marcas detectadas`}
         </h3>
         <div className="relative">
           <img 
             src={currentFrame}
             alt="Frame actual" 
             className="w-full rounded-lg"
           />
         </div>
       </div>
     )}

     {downloadUrl && fileType === 'video' && (
       <div className="mt-6 bg-white p-6 rounded-lg shadow-md">
         <h3 className="text-xl font-semibold mb-4">Resultado final</h3>
         <a 
           href={downloadUrl}
           download
           className="inline-flex items-center px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 transition"
         >
           <Download className="mr-2" />
           Descargar Video
         </a>
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

export default BrandsDetection;