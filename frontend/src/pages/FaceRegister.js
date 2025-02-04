import React, { useState } from "react";
import Webcam from "react-webcam";
import { Loader2, Camera, CheckCircle, AlertTriangle, RotateCcw, Save } from "lucide-react";
import api from "../services/api";

function FaceRegister() {
  const [personName, setPersonName] = useState("");
  const [selectedFile, setSelectedFile] = useState(null);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [capturedImage, setCapturedImage] = useState(null);
  const [previewImage, setPreviewImage] = useState(null);
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const webcamRef = React.useRef(null);

  const videoConstraints = {
    width: 640,
    height: 480,
    facingMode: "user"
  };

  const resetForm = () => {
    setPersonName("");
    setSelectedFile(null);
    setIsCameraActive(false);
    setCapturedImage(null);
    setPreviewImage(null);
    setResult(null);
    setError(null);
  };

  const handleFilePreview = async (file) => {
    if (!file) return;
    
    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("is_preview", "true");
      
      const response = await api.post("/api/faces/register", formData, {
        responseType: 'blob'
      });

      const imageUrl = URL.createObjectURL(response.data);
      setPreviewImage(imageUrl);
      setSelectedFile(file);
      setIsCameraActive(false);
      setCapturedImage(file);
    } catch (err) {
      setError("Error al procesar la imagen. Intenta de nuevo.");
    }
  };

  const handleRegister = async () => {
    if (!personName || !capturedImage) {
      setError("Por favor, ingresa un nombre y selecciona una imagen o toma una foto.");
      return;
    }
    setIsLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append("person_name", personName);
      formData.append("file", capturedImage);

      const response = await api.post("/api/faces/register", formData);
      setResult(response.data);
      resetForm();
    } catch (err) {
      setError(err.response?.data?.error || "Error al registrar el rostro.");
    } finally {
      setIsLoading(false);
    }
  };

  const capturePhoto = React.useCallback(async () => {
    const imageSrc = webcamRef.current.getScreenshot();
    if (imageSrc) {
      try {
        const response = await fetch(imageSrc);
        const blob = await response.blob();
        const file = new File([blob], `${personName || 'capture'}_${Date.now()}.jpg`, { type: 'image/jpeg' });
        await handleFilePreview(file);
      } catch (err) {
        setError("Error al capturar la foto. Intenta de nuevo.");
      }
    }
  }, [webcamRef, personName]);

  const retakePhoto = () => {
    setPreviewImage(null);
    setCapturedImage(null);
    setSelectedFile(null);
    setIsCameraActive(true);
  };

  return (
    <div className="max-w-4xl mx-auto p-6 bg-gray-50">
      <h2 className="text-3xl font-bold text-gray-800 mb-6">Registro de Rostros</h2>

      {error && (
        <div className="mb-4 bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
          <AlertTriangle className="inline-block mr-2" />
          {error}
        </div>
      )}

      {result && (
        <div className="fixed top-4 left-1/2 -translate-x-1/2 bg-green-100 border border-green-400 text-green-700 px-6 py-4 rounded-lg shadow-lg z-50 flex items-center gap-3 animate-fade-in">
          <CheckCircle className="w-6 h-6" />
          <div className="flex flex-col">
            <span className="font-semibold">¡Usuario registrado correctamente!</span>
            <span className="text-sm">Nombre: {result.name}</span>
            <span className="text-xs opacity-75">Guardado en: {result.image_path}</span>
          </div>
        </div>
      )}

      <div className="bg-white p-6 rounded-lg shadow-md space-y-6">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Nombre de la persona:
          </label>
          <input
            type="text"
            value={personName}
            onChange={(e) => setPersonName(e.target.value)}
            className="w-full p-2 border rounded focus:ring-2 focus:ring-blue-500"
            placeholder="Ej: Juan Pérez"
          />
        </div>

        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Subir imagen:
            </label>
            <input
              type="file"
              accept="image/*"
              onChange={(e) => e.target.files[0] && handleFilePreview(e.target.files[0])}
              className="w-full p-2 border rounded"
            />
          </div>

          <div className="border-t pt-4">
            <div className="flex gap-4">
              {!previewImage && (
                <button
                  onClick={() => setIsCameraActive(!isCameraActive)}
                  className="flex items-center gap-2 px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600"
                >
                  <Camera />
                  {isCameraActive ? "Desactivar cámara" : "Activar cámara"}
                </button>
              )}
              
              {isCameraActive && !previewImage && (
                <button
                  onClick={capturePhoto}
                  disabled={isLoading}
                  className="flex items-center gap-2 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:opacity-50"
                >
                  {isLoading ? <Loader2 className="animate-spin" /> : <Camera />}
                  {isLoading ? "Procesando..." : "Capturar foto"}
                </button>
              )}

              {previewImage && (
                <>
                  <button
                    onClick={retakePhoto}
                    className="flex items-center gap-2 px-4 py-2 bg-yellow-500 text-white rounded hover:bg-yellow-600"
                  >
                    <RotateCcw />
                    Repetir foto
                  </button>
                  <button
                    onClick={handleRegister}
                    disabled={isLoading || !personName}
                    className="flex items-center gap-2 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:opacity-50"
                  >
                    {isLoading ? <Loader2 className="animate-spin" /> : <Save />}
                    {isLoading ? "Registrando..." : "Registrar"}
                  </button>
                </>
              )}

              <button
                onClick={resetForm}
                className="px-4 py-2 bg-orange-500 text-white rounded hover:bg-orange-600"
              >
                Reiniciar
              </button>
            </div>

            <div className="mt-4 flex justify-center">
              <div className="relative w-[640px] h-[480px] bg-gray-100 rounded-lg overflow-hidden">
                {isCameraActive && !previewImage && (
                  <Webcam
                    ref={webcamRef}
                    audio={false}
                    screenshotFormat="image/jpeg"
                    videoConstraints={videoConstraints}
                    className="w-full h-full object-cover"
                  />
                )}
                {previewImage && (
                  <img
                    src={previewImage}
                    alt="Preview"
                    className="w-full h-full object-cover"
                  />
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default FaceRegister;