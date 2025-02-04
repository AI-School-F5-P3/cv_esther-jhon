// src/components/Layout.js


import React from "react";
import { Link, useLocation } from "react-router-dom";
import { 
  HiHome,
  HiCamera,
  HiUserGroup,
  HiUser,
  HiVideoCamera
} from "react-icons/hi";

export default function Layout({ children }) {
  const location = useLocation();

  const navItems = [
    { icon: HiHome, path: "/", label: "Inicio" },
    { icon: HiCamera, path: "/brands", label: "Detección de Marcas" },
    { icon: HiUserGroup, path: "/faces", label: "Detección de Rostros" },
    { icon: HiUser, path: "/register-face", label: "Registro de Rostro" },
    { icon: HiVideoCamera, path: "/realtime", label: "Tiempo Real" },
  ];

  return (
    <div className="min-h-screen relative flex items-center justify-center p-4">
      <video
        autoPlay
        loop
        muted
        className="absolute inset-0 w-full h-full object-cover"
      >
        <source src="/frames/video.mp4" type="video/mp4" />
        Tu navegador no soporta el elemento de video.
      </video>
      <div className="w-full max-w-6xl bg-white bg-opacity-90 rounded-xl shadow-2xl overflow-hidden flex relative z-10">
        {/* Sidebar */}
        <div className="w-20 bg-gray-800 text-white flex flex-col items-center py-8">
          {navItems.map((item) => (
            <Link
              key={item.path}
              to={item.path}
              className={`p-3 rounded-lg mb-4 hover:bg-gray-700 transition-colors duration-200 ${
                location.pathname === item.path ? "bg-gray-700" : ""
              }`}
              title={item.label}
            >
              <item.icon size={24} />
            </Link>
          ))}
        </div>

        {/* Contenido Principal */}
        <div className="flex-1 flex flex-col">
          <h1 className="text-3xl font-bold text-center py-6 bg-gray-50 border-b">
            Tu Detector 
          </h1>
          <div className="p-8 overflow-auto flex-1">
            {children}
          </div>
        </div>
      </div>
    </div>
  );
}