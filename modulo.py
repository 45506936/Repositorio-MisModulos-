import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
from numpy.random import uniform, normal
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc
)

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform
from typing import List, Tuple

class AnalisisDescriptivo:
    """
    Clase que realiza análisis descriptivo y estimación de densidades a partir de una muestra de datos.
    """

    def __init__(self, datos: List[float]) -> None:
        """
        Inicializa la clase con los datos de entrada.
        """
        self.datos = np.array(datos)

    def calculo_de_media(self) -> float:
        """
        Calcula la media de los datos.
        """
        return sum(self.datos) / len(self.datos)

    def calculo_de_mediana(self) -> float:
        """
        Calcula la mediana de los datos.
        """
        return np.median(self.datos)

    def genera_histograma(self, h: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Genera un histograma con ancho de clase h.
        """
        bins = np.arange(min(self.datos) - h/2, max(self.datos) + h, h)
        fr_abs = np.zeros(len(bins) - 1)

        # Contar frecuencias absolutas
        for valor in self.datos:
            for i_bins in range(len(bins) - 1):
                if bins[i_bins] <= valor < bins[i_bins + 1]:
                    fr_abs[i_bins] += 1
                    break

        fr_rel = fr_abs / (len(self.datos) * h)
        return bins, fr_rel

    def evalua_histograma(self, h: float, x: List[float]) -> np.ndarray:
        """
        Evalúa el histograma generado en los puntos dados por x.
        """
        bins, frec = self.genera_histograma(h)
        resultados = np.zeros(len(x))

        for i in range(len(x)):
            for i_bins in range(len(bins) - 1):
                if bins[i_bins] <= x[i] < bins[i_bins + 1]:
                    resultados[i] = frec[i_bins]
                    break
        return resultados

    def kernel_gaussiano(self, x: float) -> float:
        """
        Devuelve el valor del kernel gaussiano estándar en x.
        """
        return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)

    def kernel_uniforme(self, x: float) -> float:
        """
        Devuelve el valor del kernel uniforme en x.
        """
        return 1.0 if abs(x) <= 0.5 else 0.0

    def kernel_cuadratico(self, x: float) -> float:
        """
        Devuelve el valor del kernel cuadrático en x.
        """
        return 0.75 * (1 - x**2) if abs(x) <= 1 else 0.0

    def kernel_triangular(self, x: float) -> float:
        """
        Devuelve el valor del kernel triangular en x.
        """
        if -1 < x < 0:
            return 1 + x
        elif 0 < x < 1:
            return 1 - x
        return 0.0

    def mi_densidad(self, x: List[float], h: float, kernel: str) -> np.ndarray:
        """
        Estima la densidad.
        
        Parámetros:
        - x: Puntos donde se evaluará la densidad.
        - h: Ancho de la ventana.
        - kernel: Tipo de kernel a usar: "gaussiano", "uniforme", "cuadratico", "triangular".
        """
        densidad = np.zeros(len(x))
        for i in range(len(x)):
            for j in range(len(self.datos)):
                u = (x[i] - self.datos[j]) / h  # Normalización de la distancia
                if kernel == "gaussiano":
                    densidad[i] += self.kernel_gaussiano(u)
                elif kernel == "uniforme":
                    densidad[i] += self.kernel_uniforme(u)
                elif kernel == "cuadratico":
                    densidad[i] += self.kernel_cuadratico(u)
                elif kernel == "triangular":
                    densidad[i] += self.kernel_triangular(u)

        densidad = densidad / (len(self.datos) * h)
        return densidad

    def miqqplot(self) -> None:
        """
        Genera un QQ-Plot (gráfico de cuantiles teóricos vs. cuantiles muestrales estandarizados).
        """
        media = np.mean(self.datos)
        desvio = np.std(self.datos)
        x_ord = np.sort(self.datos)
        x_ord_s = (x_ord - media) / desvio  # estandarización
        n = len(self.datos)

        cuantiles_teoricos = [norm.ppf(p / (n + 1)) for p in range(1, n + 1)]

        plt.scatter(cuantiles_teoricos, x_ord_s, color='blue', marker='o')
        plt.xlabel('Cuantiles teóricos')
        plt.ylabel('Cuantiles muestrales')
        plt.plot(cuantiles_teoricos, cuantiles_teoricos, linestyle='-', color='red')  # línea identidad
        plt.title('QQ-Plot')
        plt.grid(True)
        plt.show()

  pass

class GeneradoraDeDatos:
    """
    Clase para generar datos de diferentes distribuciones y calcular funciones de densidad teóricas.
    """

    def __init__(self, n: int) -> None:
        """
        Inicializa la clase con la cantidad de datos a generar.
        """
        self.n = n

    def generar_datos_dist_norm(self, media: float, desvio: float) -> np.ndarray:
        """
        Genera datos a partir de una distribución normal con media y desvío dado.
        """
        return np.random.normal(media, desvio, self.n)

    def pdf_norm(self, x: np.ndarray, media: float, desvio: float) -> np.ndarray:
        """
        Calcula la densidad normal teórica en los puntos x.
        """
        return norm.pdf(x, media, desvio)

    def generar_datos_bs(self) -> np.ndarray:
        """
        Genera una mezcla de distribuciones normales y una uniforme, útil para pruebas de estimación de densidad.
        """
        u = np.random.uniform(size=self.n)
        y = u.copy()

        # 50% serán normales estándar
        y[u > 0.5] = np.random.normal(0, 1, size=np.sum(u > 0.5))

        # 50% mezcla de normales con medias crecientes
        for j in range(5):
            idx = (u > j * 0.1) & (u <= (j + 1) * 0.1)
            y[idx] = np.random.normal(j / 2 - 1, 0.1, size=np.sum(idx))
        return y

  
    def f_bs(x: float) -> float:
        """
        Función de densidad teórica correspondiente a la mezcla de distribuciones usada en `generar_datos_bs`.
        """
        mezcla = 0.5 * norm.pdf(x, 0, 1) + 0.1 * sum([norm.pdf(x, j / 2 - 1, 0.1) for j in range(5)])
        return mezcla

    def generar_datos_uniforme(self, min_: float, max_: float) -> np.ndarray:
        """
        Genera datos de una distribución uniforme en el intervalo [min_, max_].
        """
        return np.random.uniform(min_, max_, self.n)

    def distribucion_uniforme_teorica(self, a: float, b: float, num_puntos: int = 1000) -> np.ndarray:
        """
        Calcula la densidad teórica de una distribución uniforme en un rango extendido para visualización.
        """
        x = np.linspace(a - 1, b + 1, num_puntos)
        y = uniform.pdf(x, loc=a, scale=b - a)
        return y

  pass


class Regresion:
    """
    Clase base para modelos de regresión lineal y logística.
    """

    def __init__(self, x, y):
        """
        Inicializa el objeto con variables predictoras x y respuesta y.

        Parámetros:
        - x: DataFrame con las variables independientes.
        - y: Serie con la variable dependiente.
        """
        self.x = x
        self.y = y
        self.resultado

    def ajustar_modelo_lineal(self):
        """
        Ajusta un modelo de regresión lineal utilizando OLS (mínimos cuadrados ordinarios).
        """
        X = sm.add_constant(self.x)
        modelo = sm.OLS(self.y, X)
        self.resultado = modelo.fit()

    def ajustar_modelo_logistico(self):
        """
        Ajusta un modelo de regresión logística binaria.
        """
        X = sm.add_constant(self.x)
        modelo = sm.Logit(self.y, X)
        self.resultado = modelo.fit()
  pass


class RegresionLineal(Regresion):
    """
    clase para análisis y visualización específica de regresión lineal.
    """

    def graficar_dispersión_y_recta(self):
        """
        Grafica la dispersión de cada variable independiente frente a la respuesta,
        junto con la recta de regresión estimada manteniendo el resto de variables en cero.
        """
        if self.resultado is None:
            self.ajustar_modelo_lineal()

        for col in self.x.columns:
            plt.scatter(self.x[col], self.y, label='Datos')

            X_plot = self.x.copy()

            # Fijar las otras variables en cero para aislar el efecto de `col`
            for other_col in X_plot.columns:
                if other_col != col:
                    X_plot[other_col] = 0

            X_plot = sm.add_constant(X_plot)
            y_pred = self.resultado.predict(X_plot)

            plt.plot(self.x[col], y_pred, color='red', label='Recta ajustada')
            plt.xlabel(col)
            plt.ylabel('Respuesta')
            plt.title(f'{col} vs Respuesta')
            plt.legend()
            plt.grid(True)
            plt.show()

    def coeficiente_correlacion(self) -> pd.Series:
        """
        Calcula el coeficiente de correlación de Pearson entre cada predictor y la respuesta.
        """
        return self.x.apply(lambda col: col.corr(self.y))

    def r2_y_ajustado(self):
        """
        Devuelve el R² y R² ajustado del modelo.
        """
        if self.resultado is None:
            self.ajustar_modelo_lineal()
        return {
            "R2": self.resultado.rsquared,
            "R2_ajustado": self.resultado.rsquared_adj
        }

    def residuos(self):
        """
        Devuelve los residuos del modelo ajustado.
        """
        if self.resultado is None:
            self.ajustar_modelo_lineal()
        return self.resultado.resid

    def analisis_residuos(self):
        """
        Muestra el QQ plot de los residuos y el gráfico de residuos vs. valores predichos.
        """
        if self.resultado is None:
            self.ajustar_modelo_lineal()

        residuos = self.resultado.resid
        predichos = self.resultado.fittedvalues

        # QQ plot
        sm.qqplot(residuos, line='45')
        plt.title("QQ Plot de los residuos")
        plt.grid(True)
        plt.show()

        # Residuos vs valores predichos
        plt.scatter(predichos, residuos, alpha=0.7)
        plt.axhline(y=0, color='red', linestyle='--')
        plt.xlabel("Valores predichos")
        plt.ylabel("Residuos")
        plt.title("Residuos vs Valores predichos")
        plt.grid(True)
        plt.show()

    def intervalos(self, alpha: float = 0.05):
        """
        Devuelve los intervalos de confianza para los coeficientes del modelo.
        """
        if self.resultado is None:
            self.ajustar_modelo_lineal()
        return self.resultado.conf_int(alpha=alpha)

    def intervalo_prediccion(self, new_x, alpha: float = 0.05):
        """
        Calcula el intervalo de predicción para nuevas observaciones.

        Parámetros:
        - new_x: DataFrame con nuevas observaciones (sin la constante).
        - alpha: Nivel de significancia.
        """
        if self.resultado is None:
            self.ajustar_modelo_lineal()

        X_new = sm.add_constant(new_x)
        prediccion = self.resultado.get_prediction(X_new)
        intervalo = prediccion.summary_frame(alpha=alpha)

        return intervalo



class RegresionLogistica:
    """""
Este clase proporciona funciones para realizar regresión logística utilizando la librería statsmodels,
incluyendo métricas de clasificación, predicción con umbral personalizado y curva ROC con AUC.
    """

    def __init__(self, x, y):
        """
        Inicializa una instancia de la clase RegresionLogistica.
        """
      super().__init__(x, y)


    def entrenar(self, X: np.ndarray, y: np.ndarray):
        """
        Entrena el modelo de regresión logística utilizando statsmodels.

        Ejemplo:
            >>> X = np.array([[1], [2], [3], [4]])
            >>> y = np.array([0, 0, 1, 1])
            >>> model = RegresionLogistica()
            >>> model.entrenar(X, y)
        """
        X = sm.add_constant(X)  # Agregar constante
        self.modelo = sm.Logit(y, X)
        self.resultados = self.modelo.fit(disp=0)

        # Guardar estadísticas relevantes
        self.betas = self.resultados.params
        self.stderr = self.resultados.bse
        self.t_obs = self.resultados.tvalues
        self.p_valores = self.resultados.pvalues

    def predecir(self, new_x):
      miRLog = Regresion(self.x, self.y)
      res = miRLog.ajustar_modelo_logistico()
      X_new = sm.add_constant(new_x)

      return res.predict(X_new)

    def obtener_metricas(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Calcula métricas de evaluación del modelo sobre un conjunto de prueba.

            >>> model.obtener_metricas(X_test, y_test)
            {'confusion_matrix': ..., 'error': ..., 'sensibilidad': ..., 'especificidad': ...}
        """
        y_pred = self.predecir(X_test)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        error = (fp + fn) / (tn + fp + fn + tp)
        sensibilidad = tp / (tp + fn) if (tp + fn) > 0 else 0
        especificidad = tn / (tn + fp) if (tn + fp) > 0 else 0

        return {
            'confusion_matrix': cm,
            'error': error,
            'sensibilidad': sensibilidad,
            'especificidad': especificidad
        }

    def curva_roc(self, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Calcula y grafica la curva ROC del modelo.

        """
        X = sm.add_constant(X_test)
        y_prob = self.resultados.predict(X)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_score = auc(fpr, tpr)

        # Graficar la curva
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
        plt.plot([0, 1], [0, 1], 'k--', label='Azar')
        plt.xlabel('Tasa de falsos positivos (FPR)')
        plt.ylabel('Tasa de verdaderos positivos (TPR)')
        plt.title('Curva ROC')
        plt.legend()
        plt.grid(True)
        plt.show()

        return fpr, tpr, auc_score

    def resumen(self):
        """
        Imprime el resumen del modelo ajustado.
        """
        print(self.resultados.summary())


class Cualitativas:
    def __init__(self, observados, probabilidades):
        """
        Test de bondad de ajuste mediante Método de Chi Cuadrado
        Entradas: observados = datos muestrales
                  probabilidades teóricas bajo la hipótesis nula
        """
        self.observados = observados
        self.p = probabilidades
        self.n = sum(observados)  # Sumar los observados para calcular los esperados

        # Verificar que las probabilidades sumen 1
        if not np.isclose(sum(self.p), 1):
            raise ValueError("Las probabilidades deben ser una lista de números que sumen 1.")

        # Verificar que las longitudes de observados y probabilidades sean iguales
        if len(self.p) != len(self.observados):
            raise ValueError("Probabilidades y observados deben tener igual tamaño.")

        # Calcular los esperado
        self.esperados = [self.n * p for p in probabilidades]

    def chi_cuadrado(self):
        """
        Calcula el estadístico Chi Cuadrado
        """
        self.estadistico = np.sum((np.array(self.observados) - np.array(self.esperados)) ** 2 / np.array(self.esperados))
        return self.estadistico

    def percentil(self, alpha):
        """
        Calcula el valor crítico del estadístico (el cual no debe superar)
        En la entrada se debe agregar como parámetro el alfa correspondiente.
        """
        self.df = len(self.observados) - 1  # grados de libertad
        percentil_chi2 = chi2.ppf(q=1 - alpha, df=self.df)
        return percentil_chi2

    def p_valor(self):
        """
        Calcula el p_valor correspondiente con significancia 1-alfa del test de hipótesis
        H_0 = La distribución de los datos sigue la probabilidad teórica
        H_1 = La distribución de los datos NO sigue la probabilidad teórica
        """
        p_valor = 1 - chi2.cdf(self.estadistico, self.df)
        return p_valor
class nuevafuncion(probandogit):
  pass
