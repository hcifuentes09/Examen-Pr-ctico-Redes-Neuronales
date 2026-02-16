# Examen Práctico – Redes Neuronales

## Parte I: Exploración del Conjunto de Datos

### 1. Exploración inicial del dataset

**a) Número de observaciones (n):**
El conjunto de datos California Housing contiene **20640 observaciones**.

**b) Número de características por observación:**
Cada observación del dataset está descrita por **8 características numéricas.**

**c) Vector de características de una observación:**
Siguiendo la convención introducida en el **Capítulo 2 del material del curso (Arquitectura de las redes neuronales)**, las características de una observación se representan mediante un vector columna \(\mathbf{x}\), cuyos componentes corresponden a cada una de las variables de entrada. En este caso, al tratarse de 8 características, el vector se escribe como:

\[
\mathbf{x} =
\begin{pmatrix}
x_1 \\
x_2 \\
x_3 \\
x_4 \\
x_5 \\
x_6 \\
x_7 \\
x_8
\end{pmatrix}
\]

### 2. Características del conjunto de datos

El conjunto de datos California Housing está descrito por un vector de características \(\mathbf{x}\) compuesto por 8 variables, cuya definición se obtiene a partir de `california.feature_names` y la descripción oficial del dataset (`california.DESCR`) [2]:
- **MedInc**: ingreso medio de los hogares en el distrito.
- **HouseAge**: edad media de las viviendas en el distrito.
- **AveRooms**: número promedio de habitaciones por vivienda.
- **AveBedrms**: número promedio de dormitorios por vivienda.
- **Population**: población total del distrito.
- **AveOccup**: número promedio de ocupantes por vivienda.
- **Latitude**: latitud geográfica del distrito.
- **Longitude**: longitud geográfica del distrito.
Estas variables constituyen el conjunto de características de entrada que alimentan al perceptrón, tal como se describe en el Capítulo 2 del material del curso [1].

### 3. Variable objetivo

La variable que se desea predecir es la etiqueta \(y\), correspondiente a **MedHouseVal**, que representa el valor mediano de las viviendas en cada distrito censal
Las unidades de esta variable están expresadas en **centenas de miles de dólares estadounidenses (USD)**, según la documentación oficial del dataset California Housing en scikit-learn [2].

### 4. Tipo de problema

Según la taxonomía presentada en el **Capítulo 2 del material del curso**, este es un problema de **regresión**, ya que la variable objetivo \(y\) corresponde a un valor numérico continuo.
El objetivo del modelo no es asignar una etiqueta discreta a cada observación, sino **estimar un valor real**, lo cual caracteriza a los problemas de regresión. Tal como se describe en el capítulo, en este tipo de tareas la salida del modelo pertenece a la recta real y suele entrenarse utilizando funciones de pérdida como el error cuadrático medio (MSE) [1].

### 5. Normalización de los datos

A partir de la división del conjunto de datos en entrenamiento y prueba, se aplica un proceso de normalización utilizando `StandardScaler`, tal como se muestra en el código proporcionado.

**d) Importancia de la normalización:**  
La normalización de las características es fundamental para el entrenamiento estable de una red neuronal, ya que el proceso de descenso del gradiente es sensible a la escala de las variables de entrada, como se discute en el Capítulo 2 del material del curso [1]. Si las características presentan magnitudes muy diferentes, la función de pérdida adquiere una geometría alargada, lo que provoca oscilaciones y una convergencia lenta del algoritmo.

**e) ¿Qué hace exactamente StandardScaler?**
Según la documentación oficial de scikit-learn, `StandardScaler` estandariza cada característica restando su media y dividiéndola por su desviación estándar [3]:

\[
x_{\text{norm}} = \frac{x - \mu}{\sigma}
\]

donde \(\mu\) es la media de la característica y \(\sigma\) su desviación estándar, calculadas sobre el conjunto de entrenamiento.

**f) Uso de fit\_transform y transform:**  
Se utiliza `fit_transform` en el conjunto de entrenamiento para ajustar los parámetros de normalización y aplicar la transformación. En el conjunto de prueba se emplea únicamente `transform` para aplicar la misma transformación ya aprendida, evitando la filtración de información (*data leakage*) y garantizando una evaluación correcta del modelo [3].





## Referencias
[1] Material del curso. Capítulo 2: *Arquitectura de las redes neuronales*.  
[2] Scikit-learn documentation. *California Housing Dataset*.  
https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset  
[3] Scikit-learn documentation. *StandardScaler*.  
https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html



