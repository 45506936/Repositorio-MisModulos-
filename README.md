# 📊 Mi Módulo Python de Estadística

Este proyecto es un módulo en Python que agrupa herramientas estadísticas para análisis de datos, generación de muestras, ajuste de modelos de regresión y evaluación de modelos clasificadores.

Diseñado para quienes desean realizar análisis exploratorios y aplicar modelos estadísticos de manera clara, visual y didáctica.

---

## 🧠 Descripción General

Este módulo proporciona clases fáciles de usar para:

- Explorar y visualizar datos estadísticos.
- Generar muestras simuladas desde diversas distribuciones.
- Estimar densidades de probabilidad mediante kernels.
- Ajustar modelos de regresión lineal y logística con `statsmodels`.
- Evaluar modelos clasificadores con métricas clave como sensibilidad, especificidad y AUC.

---

## 🚀 Funcionalidades Principales

### 🔹 `AnalisisDescriptivo`
Clase para análisis exploratorio de datos. Permite:

- Calcular medidas estadísticas (media, varianza, cuartiles).
- Graficar histogramas y densidades kernel.
- Visualizar QQ-plots para chequear normalidad.

---

### 🔹 `GeneradoraDeDatos`
Clase para simular datos desde distribuciones teóricas:

- Normales (una o varias poblaciones).
- Uniformes.
- Mezclas de normales.
- Incluye graficación de histogramas y densidades teóricas.

---

### 🔹 `Regresion` (Clase base)
Clase abstracta para modelos de regresión lineal o logística. Encapsula:

- Preparación de datos con constante.
- División en conjunto de entrenamiento y prueba.
- Cálculo de betas, errores estándar, valores t y p-valores.
- Evaluación general del ajuste del modelo.

---

### 🔹 `RegresionLineal` (hereda de `Regresion`)
Ajuste de modelos lineales simples y múltiples. Ofrece:

- Gráfico de dispersión y recta de regresión (para regresión simple).
- Métricas del modelo: R², residuos, intervalos de confianza.
- Resumen detallado del modelo ajustado.

---

### 🔹 `RegresionLogistica` (hereda de `Regresion`)
Implementa regresión logística binaria. Además del ajuste:

- Permite predicciones con umbral ajustable.
- Calcula matriz de confusión, sensibilidad, especificidad y error.
- Grafica la curva ROC y calcula el área bajo la curva (AUC).

---

### 🔹 `RegresionCualitativa`
Clase para evaluar el desempeño de cualquier clasificador binario. Acepta las predicciones y probabilidades para:

- Calcular matriz de confusión.
- Determinar precisión, sensibilidad, especificidad y error.
- Graficar curva ROC y computar AUC.

---


