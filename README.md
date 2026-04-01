# Deep_Learning_OCR_TensorFlow_Python3.12
Ejemplo práctico: reconocimiento básico de números y texto con TensorFlow/Keras y OpenCV

# Reconocimiento de Texto y Números con TensorFlow y OpenCV 🤖🔢

Este repositorio contiene una aplicación avanzada de **Percepción Computacional** que implementa un modelo de **Deep Learning** (Redes Neuronales Convolucionales - CNN) para el reconocimiento híbrido de frases textuales y dígitos manuscritos.

## 💻 Especificaciones del Envono de Desarrollo
Para asegurar la estabilidad de las librerías de Inteligencia Artificial, se han definido las siguientes versiones obligatorias:

* **Sistema Operativo:** Windows 11 Home (Versión 25H2) 🪟
* **Lenguaje:** **Python 3.12.0 (64-bit)** 🐍 [Versión requerida para TF/Keras]
* **IDE:** Visual Studio Code
* **Librerías:** TensorFlow/Keras, OpenCV, NumPy, Matplotlib

## 📁 Estructura del Repositorio
* `reconocedor_texto_numeros_tensorflow_opencv_corregido_v3.py`: Pipeline completo de arquitectura neuronal, entrenamiento estratificado e inferencia.
* `imagen_ejemplo_3.jpg`: Imagen fuente para pruebas de reconocimiento híbrido.

## 🚀 Metodología de Procesamiento (V3)
El flujo de trabajo implementado en esta versión corregida sigue estos pasos de ingeniería:

1. **Arquitectura CNN:** Diseño de una red neuronal profunda con capas de convolución, pooling y unidades de abandono (dropout) para mitigar el sobreajuste.
2. **Entrenamiento Estratificado:** División de datos que asegura una representación equitativa de clases, fundamental para el reconocimiento de caracteres balanceado.
3. **Pipeline de OCR:**
   - **Segmentación:** OpenCV detecta y extrae Regiones de Interés (ROI).
   - **Clasificación:** Inferencia mediante el modelo entrenado con un umbral de confianza de **0.55**.
4. **Validación de Resultados:** Generación de métricas de exactitud (Accuracy) tanto en entrenamiento como en validación.

## 🛠️ Instalación y Uso
Es indispensable utilizar la versión de Python especificada para evitar conflictos de bits:

1. **Instalar dependencias:**
   ```powershell
   pip install tensorflow opencv-python numpy matplotlib
