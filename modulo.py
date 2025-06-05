import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
from numpy.random import uniform, normal

class AnalisisDescriptivo:
  def __init__(self, datos):
    self.datos = np.array(datos)

  def calculo_de_media(self):
    return sum(self.datos) / len(self.datos)

  def calculo_de_mediana(self):
    return np.median(self.datos)

  def genera_histograma(self, h):
    bins = np.arange(min(self.datos) - h/2 , max(self.datos) + h, h)
    fr_abs = np.zeros(len(bins) - 1)

    for valor in self.datos:
      for i_bins in range(len(bins) - 1):
        if bins[i_bins] <= valor < bins[i_bins + 1]:
          fr_abs[i_bins] += 1
          break

    fr_rel = fr_abs / (len(self.datos) * h)
    return bins, fr_rel

  def evalua_histograma(self, h, x):
    bins, frec = self.genera_histograma(h)
    resultados = np.zeros(len(x))

    for i in range(len(x)):
      for i_bins in range(len(bins) - 1):
        if bins[i_bins] <= x[i] < bins[i_bins + 1]:
          resultados[i] = frec[i_bins]
          break
    return resultados
  def kernel_gaussiano(self, x):
    # Kernel gaussiano estándar
    valor_kernel_gaussiano = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)
    return valor_kernel_gaussiano

  def kernel_uniforme(self, x):
    # Kernel uniforme
    if abs(x) <= 0.5:
      valor_kernel_uniforme = 1
    else:
      valor_kernel_uniforme = 0
    return valor_kernel_uniforme

  def kernel_cuadratico(self, x):
    valor_kernel_cuadratico = 3/4 * (1-x**2) if np.abs(x) <= 1 else 0
    return valor_kernel_cuadratico

  def kernel_triangular (self,x):
    if -1<x<0:
      valor_kernel_triangular = 1+x
    elif 0<x<1:
      valor_kernel_triangular = 1-x
    else:
      valor_kernel_triangular = 0

    return valor_kernel_triangular

  def mi_densidad(self, x, h, kernel):
    # x: Puntos en los que se evaluará la densidad
    # data: Datos
    # h: Ancho de la ventana (bandwidth)
    densidad = np.zeros(len(x))
    for i in range(len(x)):
      for j in range(len(self.datos)):
        u = (x[i] - self.datos[j]) / h
#Normalización de la distancia Se calcula 𝑢 u como la distancia normalizada entre el punto de evaluación x[i] y cada dato 𝑑 𝑎 𝑡 𝑎 [ 𝑗 ]. Este valor 𝑢 u es el argumento que se pasa a la función kernel, y su propósito es medir cuán lejos está 𝑑 𝑎 𝑡 𝑎 [ 𝑗 ] del punto 𝑥x[i] en unidades de ℎ.
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
  def miqqplot(self):

    media = np.mean(self.datos)
    desvio = np.std(self.datos)

    x_ord = np.sort(self.datos)
    x_ord_s = (x_ord - media) / desvio
    n = len(self.datos)

    cuantiles_teoricos = []

    for p in range(1,n+1):
      pp = p/(n+1) #convierte lista en decimales
      valor_cuantil = norm.ppf(pp)
      cuantiles_teoricos.append(valor_cuantil)


    plt.scatter(cuantiles_teoricos, x_ord_s, color='blue', marker='o')
    plt.xlabel('Cuantiles teóricos')
    plt.ylabel('Cuantiles muestrales')
    plt.plot(cuantiles_teoricos, cuantiles_teoricos, linestyle='-', color='red')
    plt.show()
  pass

class GeneradoraDeDatos:
  def __init__(self, n):
    self.n=n

  def generar_datos_dist_norm(self, media, desvio):
    return np.random.normal(media, desvio, self.n)

  def pdf_norm(self, x, media, desvio):
    p = norm.pdf(x, media, desvio)
    return p

  def generar_datos_bs(self):
    u = np.random.uniform(size=(self.n,))
    y = u.copy()
    ind = np.where(u > 0.5)[0]
    y[ind] = np.random.normal(0, 1, size=len(ind))
    for j in range(5):
        ind = np.where((u > j * 0.1) & (u <= (j+1) * 0.1))[0]
        y[ind] = np.random.normal(j/2 - 1, 1/10, size=len(ind))
    return y

  def f_bs(x):
    funcion = 0.5*norm.pdf(x,0,1)+0.1*sum([norm.pdf(x,j/2-1,0.1) for j in range(5)])
    return funcion

  def generar_datos_uniforme(self, min, max):
    datos = np.random.uniform(min, max, self.n)
    return datos

  def distribucion_uniforme_teorica(self, a, b, num_puntos=1000):
    x = np.linspace(a - 1, b + 1, num_puntos)  # Extiende el rango para mejor visualización
    y = uniform.pdf(x, loc=a, scale=b-a)
    return y
  pass


class Regresion:
  def __init__(self,x,y):
    self.x = x
    self.y = y

  def ajustar_modelo_lineal(self):
    X = sm.add_constant(self.x)
    model = sm.OLS(self.y,X)
    result = model.fit()

  def ajustar_modelo_logistico(self):
    X = sm.add_constant(self.x)
    modelo = sm.Logit(self.y, X)
    result = modelo.fit()
  pass


class RegresionLineal(Regresion):
    def graficar_dispersión_y_recta(self):
        if self.resultado is None:
            self.ajustar_modelo_lineal()

        for col in self.x.columns:
            plt.scatter(self.x[col], self.y, label='Datos')

            X_plot = self.x.copy()
            for other_col in X_plot.columns:
                if other_col != col:
                    X_plot[other_col] = 0  # fijamos las demás en cero

            X_plot = sm.add_constant(X_plot)
            y_pred = self.resultado.predict(X_plot)

            plt.plot(self.x[col], y_pred, color='red', label='Recta ajustada')
            plt.xlabel(col)
            plt.ylabel('Respuesta')
            plt.title(f'{col} vs Respuesta')
            plt.legend()
            plt.show()

    def coeficiente_correlacion(self):
        return self.x.apply(lambda col: col.corr(self.y))

    def r2_y_ajustado(self):
        if self.resultado is None:
            self.ajustar_modelo_lineal()
        return {
            "R2": self.resultado.rsquared,
            "R2_ajustado": self.resultado.rsquared_adj
        }

    def residuos(self):
        if self.resultado is None:
            self.ajustar_modelo_lineal()
        return self.resultado.resid

    def analisis_residuos(self):
        if self.resultado is None:
            self.ajustar_modelo_lineal()
        residuos = self.resultado.resid
        predichos = self.resultado.fittedvalues

        # QQ plot
        sm.qqplot(residuos, line='45')
        plt.title("QQ Plot de los residuos")
        plt.show()

        # Residuos vs valores predichos
        plt.scatter(predichos, residuos)
        plt.axhline(y=0, color='red', linestyle='--')
        plt.xlabel("Valores predichos")
        plt.ylabel("Residuos")
        plt.title("Residuos vs Valores predichos")
        plt.show()

    def intervalos(self, alpha=0.05):
        if self.resultado is None:
            self.ajustar_modelo_lineal()
        return self.resultado.conf_int(alpha=alpha)

    def intervalo_prediccion(self, new_x, alpha=0.05):
      if self.resultado is None:
         self.ajustar_modelo_lineal()

      X_new = sm.add_constant(new_x)
      prediccion = self.resultado.get_prediction(X_new)
      intervalo = prediccion.summary_frame(alpha=alpha)

      return intervalo

class RegresionLogistica(Regresion):
  pass