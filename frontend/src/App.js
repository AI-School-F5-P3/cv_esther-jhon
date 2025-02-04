// src/App.js
import React from "react";
import './index.css';
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";

import Layout from "./components/Layout";
import BrandsDetection from "./pages/BrandsDetection";
import FacesDetection from "./pages/FacesDetection";
import FaceRegister from "./pages/FaceRegister";
import RealtimeDetection from "./pages/RealtimeDetection";

function App() {
  return (
    <Router>
      <Routes>
        <Route
          path="/"
          element={
            <Layout>
              <h1 className="text-4xl font-bold mb-6 text-gray-800">Bienvenido a Tu Detector Favorito</h1>
              <p className="text-xl text-gray-600">Explora las diferentes funcionalidades de detección de objetos y rostros.</p>
            </Layout>
          }
        />
        <Route
          path="/brands"
          element={
            <Layout>
              <BrandsDetection />
            </Layout>
          }
        />
        <Route
          path="/faces"
          element={
            <Layout>
              <FacesDetection />
            </Layout>
          }
        />
        <Route
          path="/register-face"
          element={
            <Layout>
              <FaceRegister />
            </Layout>
          }
        />
        <Route
          path="/realtime"
          element={
            <Layout>
              <RealtimeDetection />
            </Layout>
          }
        />
      </Routes>
    </Router>
  );
}

export default App;