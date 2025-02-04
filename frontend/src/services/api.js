// src/services/api.js
import axios from 'axios';

const baseURL = 'http://localhost:8005';

const api = axios.create({
  baseURL,
  // Removemos el header Content-Type por defecto para que axios lo establezca automáticamente
  // cuando se usa FormData
});

// Interceptor para manejar errores
api.interceptors.response.use(
  response => response,
  error => {
    console.error('Error completo:', error);
    
    if (error.response) {
      // Si la respuesta es un Blob, necesitamos leerlo
      if (error.response.data instanceof Blob) {
        return new Promise((resolve, reject) => {
          const reader = new FileReader();
          reader.onload = () => {
            try {
              const errorData = JSON.parse(reader.result);
              error.response.data = errorData;
              reject(error);
            } catch (e) {
              reject(error);
            }
          };
          reader.onerror = () => reject(error);
          reader.readAsText(error.response.data);
        });
      }
      
      console.error('Error de respuesta:', error.response.data);
      return Promise.reject(error);
    } 
    
    if (error.request) {
      console.error('Error de solicitud:', error.request);
      return Promise.reject(new Error('No se pudo conectar con el servidor'));
    }
    
    console.error('Error:', error.message);
    return Promise.reject(error);
  }
);

export default api;