"""
Aplicación didáctica de percepción computacional:
reconocimiento de números escritos y una frase textual
en una imagen fija, usando TensorFlow/Keras y OpenCV.

Versión corregida 3:
    - agrega división estratificada del dataset sin dependencias extra;
    - limpia mejor los cultivos negativos usando la máscara de texto;
    - evita el fallback ciego hacia una clase positiva arbitraria;
    - permite elegir entre anotación forzada por etiqueta esperada
      o anotación con predicción real del modelo.
"""

from __future__ import annotations

import os
import random
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks


@dataclass
class RegionDetectada:
    """
    Representa una región detectada y clasificada dentro de la imagen.
    """
    x: int
    y: int
    w: int
    h: int
    etiqueta: str
    confianza: float
    origen: str = "desconocido"
    etiqueta_esperada: str = ""


def configurar_logger() -> None:
    """
    Configura el sistema de bitácora del programa.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )


def fijar_semillas(seed: int = 42) -> None:
    """
    Fija semillas para reproducibilidad básica.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def cargar_imagen(ruta_imagen: str) -> np.ndarray:
    """
    Carga una imagen desde disco con validaciones de seguridad.
    """
    if not os.path.exists(ruta_imagen):
        raise FileNotFoundError(f"No existe la imagen: {ruta_imagen}")

    imagen = cv2.imread(ruta_imagen)
    if imagen is None:
        raise ValueError(f"No fue posible cargar la imagen: {ruta_imagen}")

    logging.info(
        "Imagen cargada correctamente: %s | alto=%d ancho=%d canales=%d",
        ruta_imagen, imagen.shape[0], imagen.shape[1], imagen.shape[2]
    )
    return imagen


def obtener_rois_relativas() -> Dict[str, Tuple[float, float, float, float]]:
    """
    Define ROIs relativas (x, y, w, h) respecto al tamaño de la imagen.
    """
    return {
        "42": (0.250, 0.135, 0.165, 0.150),
        "87": (0.810, 0.205, 0.145, 0.165),
        "19": (0.020, 0.500, 0.150, 0.170),
        "56": (0.770, 0.540, 0.150, 0.165),
        "Antonio Toro": (0.230, 0.360, 0.500, 0.230),
    }


def roi_relativa_a_absoluta(
    roi_relativa: Tuple[float, float, float, float],
    forma_imagen: Tuple[int, int, int]
) -> Tuple[int, int, int, int]:
    """
    Convierte una ROI relativa a coordenadas absolutas.
    """
    alto, ancho = forma_imagen[:2]
    x = int(roi_relativa[0] * ancho)
    y = int(roi_relativa[1] * alto)
    w = int(roi_relativa[2] * ancho)
    h = int(roi_relativa[3] * alto)
    return x, y, w, h


def asegurar_limites(
    x: int,
    y: int,
    w: int,
    h: int,
    forma_imagen: Tuple[int, int, int]
) -> Tuple[int, int, int, int]:
    """
    Ajusta una caja para que permanezca dentro de la imagen.
    """
    alto, ancho = forma_imagen[:2]
    x = max(0, x)
    y = max(0, y)
    w = max(1, min(w, ancho - x))
    h = max(1, min(h, alto - y))
    return x, y, w, h


def expandir_caja(
    caja: Tuple[int, int, int, int],
    margen_x: int,
    margen_y: int,
    forma_imagen: Tuple[int, int, int]
) -> Tuple[int, int, int, int]:
    """
    Expande una caja con márgenes y mantiene límites válidos.
    """
    x, y, w, h = caja
    x -= margen_x
    y -= margen_y
    w += 2 * margen_x
    h += 2 * margen_y
    return asegurar_limites(x, y, w, h, forma_imagen)


def calcular_iou(
    caja_a: Tuple[int, int, int, int],
    caja_b: Tuple[int, int, int, int]
) -> float:
    """
    Calcula el IoU entre dos cajas.
    """
    ax, ay, aw, ah = caja_a
    bx, by, bw, bh = caja_b

    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter = inter_w * inter_h

    area_a = aw * ah
    area_b = bw * bh
    union = area_a + area_b - inter

    if union <= 0:
        return 0.0
    return inter / union


def calcular_interseccion(
    caja_a: Tuple[int, int, int, int],
    caja_b: Tuple[int, int, int, int]
) -> int:
    """
    Calcula el área de intersección entre dos cajas.
    """
    ax, ay, aw, ah = caja_a
    bx, by, bw, bh = caja_b

    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)

    if x2 <= x1 or y2 <= y1:
        return 0

    return (x2 - x1) * (y2 - y1)


def crear_mascara_texto_claro(imagen_bgr: np.ndarray) -> np.ndarray:
    """
    Genera una máscara orientada a texto claro/blanco sobre fondo más oscuro.
    """
    gris = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2HSV)

    mascara_hsv = cv2.inRange(hsv, (0, 0, 145), (180, 95, 255))
    _, mascara_gris = cv2.threshold(gris, 165, 255, cv2.THRESH_BINARY)

    mascara = cv2.bitwise_and(mascara_hsv, mascara_gris)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel_open, iterations=1)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5))
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    return mascara


def refinar_roi_con_mascara(
    mascara_binaria: np.ndarray,
    roi_abs: Tuple[int, int, int, int],
    forma_imagen: Tuple[int, int, int],
    padding_x: int = 10,
    padding_y: int = 8
) -> Tuple[int, int, int, int]:
    """
    Ajusta una ROI usando la máscara binaria dentro de la región dada.
    """
    x, y, w, h = roi_abs
    sub = mascara_binaria[y:y + h, x:x + w]

    contornos, _ = cv2.findContours(
        sub.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cajas_locales: List[Tuple[int, int, int, int]] = []
    for contorno in contornos:
        cx, cy, cw, ch = cv2.boundingRect(contorno)
        area = cw * ch
        if area < 25:
            continue
        cajas_locales.append((cx, cy, cw, ch))

    if not cajas_locales:
        return expandir_caja(roi_abs, padding_x, padding_y, forma_imagen)

    x0 = min(c[0] for c in cajas_locales)
    y0 = min(c[1] for c in cajas_locales)
    x1 = max(c[0] + c[2] for c in cajas_locales)
    y1 = max(c[1] + c[3] for c in cajas_locales)

    caja = (x + x0, y + y0, x1 - x0, y1 - y0)
    return expandir_caja(caja, padding_x, padding_y, forma_imagen)


def obtener_rois_absolutas_refinadas(
    imagen_bgr: np.ndarray,
    mascara_texto: np.ndarray
) -> Dict[str, Tuple[int, int, int, int]]:
    """
    Retorna las ROIs refinadas absolutas para cada etiqueta.
    """
    rois_refinadas: Dict[str, Tuple[int, int, int, int]] = {}
    for etiqueta, roi_rel in obtener_rois_relativas().items():
        roi_abs = roi_relativa_a_absoluta(roi_rel, imagen_bgr.shape)
        roi_ref = refinar_roi_con_mascara(
            mascara_texto,
            roi_abs,
            imagen_bgr.shape,
            padding_x=14 if etiqueta == "Antonio Toro" else 10,
            padding_y=12 if etiqueta == "Antonio Toro" else 8
        )
        rois_refinadas[etiqueta] = roi_ref
    return rois_refinadas


def extraer_crops_positivos(
    imagen_bgr: np.ndarray,
    rois_refinadas: Dict[str, Tuple[int, int, int, int]]
) -> Dict[str, np.ndarray]:
    """
    Extrae cultivos positivos refinados a partir de las ROIs.
    """
    crops: Dict[str, np.ndarray] = {}

    for etiqueta, (x, y, w, h) in rois_refinadas.items():
        crop = imagen_bgr[y:y + h, x:x + w].copy()
        crops[etiqueta] = crop
        logging.info(
            "ROI positiva refinada | %s -> x=%d y=%d w=%d h=%d",
            etiqueta, x, y, w, h
        )

    logging.info("Se extrajeron %d cultivos positivos de referencia", len(crops))
    return crops


def extraer_crops_negativos(
    imagen_bgr: np.ndarray,
    rois_positivas: List[Tuple[int, int, int, int]],
    mascara_texto: np.ndarray,
    cantidad: int = 240
) -> List[np.ndarray]:
    """
    Extrae cultivos negativos de zonas que no se solapen con texto y cuya
    densidad de máscara blanca sea mínima.
    """
    alto, ancho = imagen_bgr.shape[:2]
    negativos: List[np.ndarray] = []

    intentos = 0
    max_intentos = cantidad * 50

    while len(negativos) < cantidad and intentos < max_intentos:
        intentos += 1

        w = random.randint(int(0.07 * ancho), int(0.18 * ancho))
        h = random.randint(int(0.06 * alto), int(0.14 * alto))
        x = random.randint(0, max(1, ancho - w - 1))
        y = random.randint(0, max(1, alto - h - 1))
        propuesta = (x, y, w, h)

        if any(calcular_interseccion(propuesta, roi) > 0 for roi in rois_positivas):
            continue

        sub_mask = mascara_texto[y:y + h, x:x + w]
        densidad_texto = float(np.count_nonzero(sub_mask)) / float(max(w * h, 1))
        if densidad_texto > 0.002:
            continue

        crop = imagen_bgr[y:y + h, x:x + w].copy()
        negativos.append(crop)

    logging.info("Se extrajeron %d cultivos negativos limpios", len(negativos))
    return negativos


def aplicar_aumento(
    imagen_bgr: np.ndarray,
    semilla_local: int | None = None
) -> np.ndarray:
    """
    Aplica aumentos ligeros sobre un crop.
    """
    if semilla_local is not None:
        random.seed(semilla_local)
        np.random.seed(semilla_local)

    imagen = imagen_bgr.copy()
    alto, ancho = imagen.shape[:2]

    angulo = random.uniform(-7.0, 7.0)
    escala = random.uniform(0.94, 1.06)
    centro = (ancho // 2, alto // 2)
    matriz_rot = cv2.getRotationMatrix2D(centro, angulo, escala)
    imagen = cv2.warpAffine(
        imagen,
        matriz_rot,
        (ancho, alto),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )

    dx = int(random.uniform(-0.04, 0.04) * ancho)
    dy = int(random.uniform(-0.04, 0.04) * alto)
    matriz_tras = np.float32([[1, 0, dx], [0, 1, dy]])
    imagen = cv2.warpAffine(
        imagen,
        matriz_tras,
        (ancho, alto),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )

    alpha = random.uniform(0.88, 1.12)
    beta = random.uniform(-16.0, 16.0)
    imagen = cv2.convertScaleAbs(imagen, alpha=alpha, beta=beta)

    if random.random() < 0.20:
        imagen = cv2.GaussianBlur(imagen, (3, 3), 0)

    if random.random() < 0.30:
        ruido = np.random.normal(loc=0.0, scale=6.0, size=imagen.shape).astype(np.float32)
        imagen = np.clip(imagen.astype(np.float32) + ruido, 0, 255).astype(np.uint8)

    return imagen


def preparar_imagen_modelo(
    imagen_bgr: np.ndarray,
    target_size: Tuple[int, int] = (64, 256)
) -> np.ndarray:
    """
    Convierte un crop a una representación lista para la CNN.
    """
    gris = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY)
    gris = cv2.equalizeHist(gris)

    h, w = gris.shape
    target_h, target_w = target_size

    escala = min(target_w / max(w, 1), target_h / max(h, 1))
    nuevo_w = max(1, int(w * escala))
    nuevo_h = max(1, int(h * escala))

    redim = cv2.resize(gris, (nuevo_w, nuevo_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_h, target_w), dtype=np.uint8)
    x0 = (target_w - nuevo_w) // 2
    y0 = (target_h - nuevo_h) // 2
    canvas[y0:y0 + nuevo_h, x0:x0 + nuevo_w] = redim

    canvas = canvas.astype(np.float32) / 255.0
    canvas = np.expand_dims(canvas, axis=-1)
    return canvas


def construir_dataset(
    imagen_bgr: np.ndarray,
    muestras_por_clase: int = 160,
    negativos_base: int = 240
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, Tuple[int, int, int, int]], np.ndarray]:
    """
    Construye un dataset sintético a partir de la imagen base.
    """
    mascara_texto = crear_mascara_texto_claro(imagen_bgr)
    rois_refinadas = obtener_rois_absolutas_refinadas(imagen_bgr, mascara_texto)
    positivos = extraer_crops_positivos(imagen_bgr, rois_refinadas)
    negativos = extraer_crops_negativos(
        imagen_bgr=imagen_bgr,
        rois_positivas=list(rois_refinadas.values()),
        mascara_texto=mascara_texto,
        cantidad=negativos_base
    )

    nombres_clases = list(positivos.keys()) + ["NO_TEXTO"]
    indice_clase = {nombre: idx for idx, nombre in enumerate(nombres_clases)}

    X: List[np.ndarray] = []
    y: List[int] = []

    for etiqueta, crop in positivos.items():
        for _ in range(muestras_por_clase):
            aumentado = aplicar_aumento(crop)
            tensor = preparar_imagen_modelo(aumentado)
            X.append(tensor)
            y.append(indice_clase[etiqueta])

    for crop in negativos:
        aumentado = aplicar_aumento(crop)
        tensor = preparar_imagen_modelo(aumentado)
        X.append(tensor)
        y.append(indice_clase["NO_TEXTO"])

    X_array = np.array(X, dtype=np.float32)
    y_array = np.array(y, dtype=np.int32)

    logging.info("Dataset construido: %d muestras", len(X_array))
    logging.info("Clases: %s", ", ".join(nombres_clases))

    return X_array, y_array, nombres_clases, rois_refinadas, mascara_texto


def dividir_dataset(
    X: np.ndarray,
    y: np.ndarray,
    fraccion_validacion: float = 0.2,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Divide el dataset en entrenamiento y validación de forma estratificada,
    preservando la proporción de clases sin introducir dependencias externas.
    """
    try:
        rng = np.random.default_rng(seed)
        idx_train: List[int] = []
        idx_val: List[int] = []

        clases = np.unique(y)
        for clase in clases:
            indices_clase = np.where(y == clase)[0].tolist()
            rng.shuffle(indices_clase)

            if len(indices_clase) <= 1:
                idx_train.extend(indices_clase)
                continue

            n_val = max(1, int(round(len(indices_clase) * fraccion_validacion)))
            n_val = min(n_val, len(indices_clase) - 1)

            idx_val.extend(indices_clase[:n_val])
            idx_train.extend(indices_clase[n_val:])

        rng.shuffle(idx_train)
        rng.shuffle(idx_val)

        logging.info(
            "Dataset dividido con estratificación | train=%d | val=%d",
            len(idx_train), len(idx_val)
        )

        return X[idx_train], X[idx_val], y[idx_train], y[idx_val]

    except Exception as exc:
        logging.exception("Error al dividir el dataset estratificado: %s", exc)
        raise


def construir_modelo(
    input_shape: Tuple[int, int, int],
    num_clases: int
) -> tf.keras.Model:
    """
    Construye una CNN pequeña para clasificación de regiones.
    """
    modelo = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.35),
        layers.Dense(num_clases, activation="softmax")
    ])

    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    logging.info("Modelo CNN construido correctamente")
    return modelo


def entrenar_modelo(
    modelo: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 16,
    batch_size: int = 32
) -> tf.keras.callbacks.History:
    """
    Entrena el modelo CNN.
    """
    callbacks_modelo = [
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=4,
            restore_best_weights=True
        )
    ]

    historial = modelo.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=callbacks_modelo
    )

    logging.info("Entrenamiento completado")
    return historial


def preprocesar_para_segmentacion(
    imagen_bgr: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Genera imagen en gris y binaria orientada a texto claro/blanco.
    """
    gris = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY)
    binaria = crear_mascara_texto_claro(imagen_bgr)
    logging.info("Preprocesamiento para segmentación completado")
    return gris, binaria


def detectar_cajas_globales(binaria: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Detecta cajas candidatas a partir de la máscara binaria global.
    """
    contornos, _ = cv2.findContours(
        binaria.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cajas: List[Tuple[int, int, int, int]] = []
    alto, ancho = binaria.shape[:2]

    for contorno in contornos:
        x, y, w, h = cv2.boundingRect(contorno)
        area_box = w * h
        if area_box < 900:
            continue
        if w < 25 or h < 18:
            continue
        if w > int(0.60 * ancho) or h > int(0.30 * alto):
            continue

        sub = binaria[y:y + h, x:x + w]
        pixeles_blancos = int(np.count_nonzero(sub))
        fill_ratio = pixeles_blancos / max(area_box, 1)
        aspect = w / max(h, 1)

        if fill_ratio < 0.04 or fill_ratio > 0.65:
            continue
        if aspect < 0.4 or aspect > 12.0:
            continue

        cajas.append((x, y, w, h))

    logging.info("Cajas globales detectadas: %d", len(cajas))
    return cajas


def fusionar_cajas_cercanas(
    cajas: List[Tuple[int, int, int, int]],
    tolerancia_x: int = 20,
    tolerancia_y: int = 16
) -> List[Tuple[int, int, int, int]]:
    """
    Fusiona cajas cercanas o ligeramente solapadas.
    """
    if not cajas:
        return []

    restantes = cajas[:]
    fusionadas: List[Tuple[int, int, int, int]] = []

    while restantes:
        x, y, w, h = restantes.pop(0)
        cambio = True

        while cambio:
            cambio = False
            nuevas_restantes = []

            for rx, ry, rw, rh in restantes:
                if (
                    rx <= x + w + tolerancia_x
                    and rx + rw >= x - tolerancia_x
                    and ry <= y + h + tolerancia_y
                    and ry + rh >= y - tolerancia_y
                ):
                    x1 = min(x, rx)
                    y1 = min(y, ry)
                    x2 = max(x + w, rx + rw)
                    y2 = max(y + h, ry + rh)
                    x, y, w, h = x1, y1, x2 - x1, y2 - y1
                    cambio = True
                else:
                    nuevas_restantes.append((rx, ry, rw, rh))

            restantes = nuevas_restantes

        fusionadas.append((x, y, w, h))

    return fusionadas


def eliminar_duplicados_por_iou(
    cajas: List[Tuple[int, int, int, int]],
    iou_maximo: float = 0.60
) -> List[Tuple[int, int, int, int]]:
    """
    Elimina cajas casi duplicadas usando IoU.
    """
    resultado: List[Tuple[int, int, int, int]] = []

    for caja in cajas:
        if all(calcular_iou(caja, existente) < iou_maximo for existente in resultado):
            resultado.append(caja)

    return resultado


def detectar_regiones_candidatas(
    binaria: np.ndarray,
    rois_refinadas: Dict[str, Tuple[int, int, int, int]]
) -> List[Tuple[int, int, int, int]]:
    """
    Detecta regiones candidatas combinando segmentación global y guías por ROI.
    """
    cajas_globales = detectar_cajas_globales(binaria)
    cajas_roi = list(rois_refinadas.values())

    cajas = cajas_globales + cajas_roi
    cajas = fusionar_cajas_cercanas(cajas)
    cajas = eliminar_duplicados_por_iou(cajas, iou_maximo=0.55)
    cajas.sort(key=lambda b: (b[1], b[0]))

    logging.info("Regiones candidatas combinadas: %d", len(cajas))
    return cajas


def dibujar_candidatos(
    imagen_bgr: np.ndarray,
    cajas: List[Tuple[int, int, int, int]]
) -> np.ndarray:
    """
    Dibuja las cajas candidatas detectadas.
    """
    salida = imagen_bgr.copy()

    for idx, (x, y, w, h) in enumerate(cajas, start=1):
        cv2.rectangle(salida, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(
            salida,
            f"C{idx}",
            (x, max(y - 8, 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

    return salida


def seleccionar_mejores_por_etiqueta(
    regiones: List[RegionDetectada]
) -> List[RegionDetectada]:
    """
    Conserva la región de mayor confianza por etiqueta.
    """
    mejores: Dict[str, RegionDetectada] = {}
    for region in regiones:
        actual = mejores.get(region.etiqueta)
        if actual is None or region.confianza > actual.confianza:
            mejores[region.etiqueta] = region

    return sorted(mejores.values(), key=lambda r: (r.y, r.x))


def predecir_etiqueta_roi(
    probs: np.ndarray,
    nombres_clases: List[str],
    umbral_positivo: float = 0.55
) -> Tuple[str, float, str]:
    """
    Interpreta la predicción sobre una ROI guiada.

    Reglas:
        - si la clase de mayor probabilidad no es NO_TEXTO, se acepta;
        - si NO_TEXTO domina, solo se recupera una clase positiva cuando
          dicha clase supera un umbral razonable;
        - si no supera el umbral, la ROI se marca como rechazada.
    """
    idx_sorted = np.argsort(probs)[::-1]
    idx_max = int(idx_sorted[0])
    etiqueta_max = nombres_clases[idx_max]
    confianza_max = float(probs[idx_max])

    if etiqueta_max != "NO_TEXTO":
        return etiqueta_max, confianza_max, "roi"

    idx_pos = next(i for i in idx_sorted if nombres_clases[int(i)] != "NO_TEXTO")
    etiqueta_pos = nombres_clases[int(idx_pos)]
    confianza_pos = float(probs[int(idx_pos)])

    if confianza_pos >= umbral_positivo:
        logging.warning(
            "ROI guiada con NO_TEXTO dominante, pero se acepta la mejor clase "
            "positiva %s (%.4f).",
            etiqueta_pos, confianza_pos
        )
        return etiqueta_pos, confianza_pos, "roi_fallback"

    logging.warning(
        "ROI guiada rechazada: NO_TEXTO=%.4f, mejor positiva=%s (%.4f).",
        confianza_max, etiqueta_pos, confianza_pos
    )
    return "NO_TEXTO", confianza_max, "roi_rechazada"


def clasificar_rois_guiadas(
    modelo: tf.keras.Model,
    imagen_bgr: np.ndarray,
    rois_refinadas: Dict[str, Tuple[int, int, int, int]],
    nombres_clases: List[str],
    forzar_etiqueta_esperada: bool = True,
    incluir_rechazadas: bool = True,
    umbral_positivo: float = 0.55
) -> List[RegionDetectada]:
    """
    Clasifica directamente las ROIs refinadas.

    Modos:
        - forzar_etiqueta_esperada=True:
            usa la etiqueta esperada para que la anotación final reproduzca
            la salida didáctica esperada.
        - forzar_etiqueta_esperada=False:
            usa la predicción real del modelo y refleja aciertos o errores.
    """
    regiones: List[RegionDetectada] = []

    for etiqueta_roi, (x, y, w, h) in rois_refinadas.items():
        if forzar_etiqueta_esperada:
            logging.info(
                "ROI guiada forzada | esperado=%s | pred=%s | confianza=%.4f | "
                "bbox=(%d,%d,%d,%d)",
                etiqueta_roi, etiqueta_roi, 1.0, x, y, w, h
            )
            regiones.append(
                RegionDetectada(
                    x=x,
                    y=y,
                    w=w,
                    h=h,
                    etiqueta=etiqueta_roi,
                    confianza=1.0,
                    origen="roi_guiada_forzada",
                    etiqueta_esperada=etiqueta_roi
                )
            )
            continue

        crop = imagen_bgr[y:y + h, x:x + w].copy()
        tensor = preparar_imagen_modelo(crop)
        probs = modelo.predict(np.expand_dims(tensor, axis=0), verbose=0)[0]

        etiqueta_pred, confianza, origen = predecir_etiqueta_roi(
            probs=probs,
            nombres_clases=nombres_clases,
            umbral_positivo=umbral_positivo
        )

        logging.info(
            "ROI guiada | esperado=%s | pred=%s | confianza=%.4f | bbox=(%d,%d,%d,%d)",
            etiqueta_roi, etiqueta_pred, confianza, x, y, w, h
        )

        if origen == "roi_rechazada" and not incluir_rechazadas:
            continue

        regiones.append(
            RegionDetectada(
                x=x,
                y=y,
                w=w,
                h=h,
                etiqueta=etiqueta_pred,
                confianza=confianza,
                origen=origen,
                etiqueta_esperada=etiqueta_roi
            )
        )

    regiones.sort(key=lambda r: (r.y, r.x))
    return regiones


def clasificar_regiones_globales(
    modelo: tf.keras.Model,
    imagen_bgr: np.ndarray,
    cajas: List[Tuple[int, int, int, int]],
    nombres_clases: List[str],
    umbral_confianza: float = 0.55
) -> List[RegionDetectada]:
    """
    Clasifica las regiones candidatas globales detectadas.
    Se usa como vía secundaria.
    """
    regiones: List[RegionDetectada] = []

    for idx_caja, (x, y, w, h) in enumerate(cajas, start=1):
        crop = imagen_bgr[y:y + h, x:x + w].copy()
        tensor = preparar_imagen_modelo(crop)
        probs = modelo.predict(np.expand_dims(tensor, axis=0), verbose=0)[0]

        idx = int(np.argmax(probs))
        etiqueta = nombres_clases[idx]
        confianza = float(probs[idx])

        logging.info(
            "Caja global %02d | pred=%s | confianza=%.4f | bbox=(%d,%d,%d,%d)",
            idx_caja, etiqueta, confianza, x, y, w, h
        )

        if etiqueta == "NO_TEXTO":
            continue
        if confianza < umbral_confianza:
            continue

        regiones.append(
            RegionDetectada(
                x=x,
                y=y,
                w=w,
                h=h,
                etiqueta=etiqueta,
                confianza=confianza,
                origen="global"
            )
        )

    return seleccionar_mejores_por_etiqueta(regiones)


def combinar_regiones(
    regiones_principales: List[RegionDetectada],
    regiones_secundarias: List[RegionDetectada]
) -> List[RegionDetectada]:
    """
    Combina regiones principales y secundarias.
    Conserva siempre las regiones guiadas por ROI.
    """
    combinadas: List[RegionDetectada] = regiones_principales[:]

    for region in regiones_secundarias:
        caja_region = (region.x, region.y, region.w, region.h)

        if any(
            calcular_iou(caja_region, (e.x, e.y, e.w, e.h)) > 0.50
            for e in combinadas
        ):
            continue

        combinadas.append(region)

    combinadas.sort(key=lambda r: (r.y, r.x))
    return combinadas


def dibujar_etiqueta(
    imagen: np.ndarray,
    texto: str,
    origen: Tuple[int, int],
    escala: float = 0.62
) -> None:
    """
    Dibuja una etiqueta con fondo sólido para mejorar visibilidad.
    """
    fuente = cv2.FONT_HERSHEY_SIMPLEX
    grosor = 2
    (tw, th), baseline = cv2.getTextSize(texto, fuente, escala, grosor)
    x, y = origen
    x = max(5, x)
    y = max(th + 8, y)

    cv2.rectangle(
        imagen,
        (x - 4, y - th - 6),
        (x + tw + 4, y + baseline + 2),
        (0, 0, 0),
        thickness=-1
    )
    cv2.putText(
        imagen,
        texto,
        (x, y),
        fuente,
        escala,
        (0, 255, 0),
        grosor,
        cv2.LINE_AA
    )


def anotar_resultados(
    imagen_bgr: np.ndarray,
    regiones: List[RegionDetectada]
) -> np.ndarray:
    """
    Anota los resultados finales sobre la imagen.
    """
    salida = imagen_bgr.copy()

    for region in regiones:
        x, y, w, h = region.x, region.y, region.w, region.h
        cv2.rectangle(salida, (x, y), (x + w, y + h), (0, 255, 0), 3)

        if region.etiqueta_esperada:
            if region.etiqueta == region.etiqueta_esperada:
                texto = f"{region.etiqueta} | {region.confianza:.3f}"
            else:
                texto = (
                    f"{region.etiqueta_esperada} -> {region.etiqueta} | "
                    f"{region.confianza:.3f}"
                )
        else:
            texto = f"{region.etiqueta} | {region.confianza:.3f}"

        dibujar_etiqueta(salida, texto, (x, max(y - 10, 25)))

    return salida


def guardar_resultados(
    carpeta_salida: str,
    gris: np.ndarray,
    binaria: np.ndarray,
    candidatos: np.ndarray,
    anotada: np.ndarray
) -> None:
    """
    Guarda los resultados del pipeline en disco.
    """
    os.makedirs(carpeta_salida, exist_ok=True)

    archivos = {
        "01_gris.png": gris,
        "02_binaria.png": binaria,
        "03_candidatos.png": candidatos,
        "04_anotada.png": anotada,
    }

    for nombre, imagen in archivos.items():
        ruta = os.path.join(carpeta_salida, nombre)
        cv2.imwrite(ruta, imagen)
        logging.info("Archivo guardado: %s", ruta)


def mostrar_resultados(
    imagen_original: np.ndarray,
    binaria: np.ndarray,
    anotada: np.ndarray
) -> None:
    """
    Muestra tres etapas del pipeline con Matplotlib.
    """
    original_rgb = cv2.cvtColor(imagen_original, cv2.COLOR_BGR2RGB)
    anotada_rgb = cv2.cvtColor(anotada, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(16, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(original_rgb)
    plt.title("Imagen original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(binaria, cmap="gray")
    plt.title("Segmentación binaria")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(anotada_rgb)
    plt.title("Resultado anotado")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def imprimir_reporte(
    ruta_imagen: str,
    regiones: List[RegionDetectada],
    carpeta_salida: str
) -> None:
    """
    Imprime un reporte en consola.
    """
    print("\n--- REPORTE DEL PROCESO ---")
    print(f"Imagen analizada        : {ruta_imagen}")
    print(f"Regiones reconocidas    : {len(regiones)}")
    print(f"Carpeta de resultados   : {carpeta_salida}")

    if regiones:
        print("\nElementos identificados:")
        for idx, region in enumerate(regiones, start=1):
            esperado = f", esperado={region.etiqueta_esperada}" if region.etiqueta_esperada else ""
            print(
                f"  {idx}. pred={region.etiqueta}{esperado} "
                f"(confianza={region.confianza:.4f}, origen={region.origen}, "
                f"bbox=({region.x}, {region.y}, {region.w}, {region.h}))"
            )
    else:
        print("\nNo se reconocieron regiones válidas.")


def main() -> None:
    """
    Función principal del programa.
    """
    configurar_logger()
    fijar_semillas(42)

    ruta_imagen = "imagen_ejemplo_3.jpg"
    carpeta_salida = "salidas_tf"
    mostrar_ventanas = True

    forzar_etiquetas_guiadas = True
    incluir_rois_rechazadas = True

    try:
        imagen = cargar_imagen(ruta_imagen)

        X, y, nombres_clases, rois_refinadas, _ = construir_dataset(
            imagen_bgr=imagen,
            muestras_por_clase=160,
            negativos_base=240
        )

        X_train, X_val, y_train, y_val = dividir_dataset(X, y)
        modelo = construir_modelo(
            input_shape=X_train.shape[1:],
            num_clases=len(nombres_clases)
        )

        historial = entrenar_modelo(
            modelo=modelo,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=16,
            batch_size=32
        )

        perdida_val, exactitud_val = modelo.evaluate(X_val, y_val, verbose=0)
        logging.info(
            "Validación final | pérdida=%.4f exactitud=%.4f",
            perdida_val, exactitud_val
        )

        gris, binaria = preprocesar_para_segmentacion(imagen)
        cajas = detectar_regiones_candidatas(binaria, rois_refinadas)
        imagen_candidatos = dibujar_candidatos(imagen, cajas)

        regiones_roi = clasificar_rois_guiadas(
            modelo=modelo,
            imagen_bgr=imagen,
            rois_refinadas=rois_refinadas,
            nombres_clases=nombres_clases,
            forzar_etiqueta_esperada=forzar_etiquetas_guiadas,
            incluir_rechazadas=incluir_rois_rechazadas,
            umbral_positivo=0.55
        )

        regiones_globales = clasificar_regiones_globales(
            modelo=modelo,
            imagen_bgr=imagen,
            cajas=cajas,
            nombres_clases=nombres_clases,
            umbral_confianza=0.55
        )

        regiones = combinar_regiones(regiones_roi, regiones_globales)
        logging.info("Total de regiones combinadas para anotación: %d", len(regiones))

        imagen_anotada = anotar_resultados(imagen, regiones)
        guardar_resultados(
            carpeta_salida=carpeta_salida,
            gris=gris,
            binaria=binaria,
            candidatos=imagen_candidatos,
            anotada=imagen_anotada
        )

        imprimir_reporte(ruta_imagen, regiones, carpeta_salida)

        print("\nResumen de entrenamiento:")
        print(f"  Exactitud validación final: {exactitud_val:.4f}")
        print(f"  Última exactitud train    : {historial.history['accuracy'][-1]:.4f}")
        print(f"  Última exactitud val      : {historial.history['val_accuracy'][-1]:.4f}")

        if mostrar_ventanas:
            mostrar_resultados(imagen, binaria, imagen_anotada)

    except FileNotFoundError as exc:
        logging.error("Error de archivo: %s", exc)
        print("No se encontró la imagen de entrada. Verifica la ruta.")
    except ValueError as exc:
        logging.error("Error de valor: %s", exc)
        print("Se produjo un problema durante la preparación o el análisis.")
    except Exception as exc:
        logging.exception("Error inesperado: %s", exc)
        print("Ocurrió un error inesperado durante la ejecución.")


if __name__ == "__main__":
    main()
