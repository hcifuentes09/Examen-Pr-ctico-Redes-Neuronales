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


## Parte 2: Arquitectura del perceptron

### 6. Incialización de parámetros

**g) ¿Por qué es conveniente inicializar los pesos con valores aleatorios pequeños en lugar de ceros?**
Es conveniente inicializar los pesos con valores aleatorios pequeños para **romper la simetría** del modelo y permitir que cada peso evolucione de manera diferente durante el entrenamiento.  
Si todos los pesos se inicializaran en cero, el gradiente sería idéntico para cada uno y el modelo no podría aprender representaciones diferenciadas.  Además, valores pequeños evitan activaciones excesivamente grandes al inicio, favoreciendo una dinámica estable del descenso del gradiente, como se describe en el **Capítulo 2 del material del curso** [1].

**h) ¿Qué forma debe tener el vector de pesos \( \mathbf{w} \)?**
Según la notación del **Capítulo 2**, el vector de pesos \( \mathbf{w} \) debe representarse como un **vector columna** de dimensión \( (n,1) \), donde \( n \) es el número de características de entrada. En este caso, como el conjunto de datos tiene **8 características**, se obtiene  
\( \mathbf{w} \in \mathbb{R}^{8 \times 1} \), lo cual se verifica al imprimir `w.shape` tras llamar a la función de inicialización [1].

**i) ¿Por qué el sesgo \( b \) se inicializa típicamente en cero mientras los pesos no?**
El sesgo \( b \) se inicializa en cero porque **no introduce problemas de simetría** como los pesos y actúa únicamente como un término de desplazamiento. Durante el entrenamiento, el sesgo se ajusta mediante descenso del gradiente junto con los pesos, permitiendo desplazar la salida del modelo sin afectar la dirección inicial del aprendizaje [1].

### 7. Implementación de la suma ponderada (propagación hacia adelante)

**j) ¿Por qué usamos \( X \mathbf{w} \) en lugar de \( \mathbf{w}^T X \)?**
Cuando \( X \) tiene forma \( (m,n) \) y el vector de pesos \( \mathbf{w} \) tiene forma \( (n,1) \),
el producto matricial \( X \mathbf{w} \) produce un vector columna de dimensión \( (m,1) \), correspondiente a una predicción por observación. La expresión \( \mathbf{w}^T X \) no es compatible dimensionalmente en este contexto. Este uso sigue directamente la formulación matricial del perceptrón presentada en el **Capítulo 2 del material del curso** [1].

**k) ¿Qué función de activación se usa en este problema de regresión? ¿Por qué?**
En este problema de regresión se utiliza una **función de activación lineal (identidad)**, es decir, no se aplica ninguna transformación no lineal a la salida. Esto se debe a que el objetivo es predecir un **valor continuo**, y la salida del modelo debe pertenecer a la recta real, tal como se describe en el **Capítulo 2** [1].

**l) Predicciones iniciales y comparación con valores reales**
Al evaluar la función de propagación hacia adelante sobre los primeros cinco ejemplos del conjunto de entrenamiento, se obtienen las siguientes predicciones iniciales:

Predicciones:
\[
[\, 0.0147,\ -0.0178,\ 0.0463,\ 0.0197,\ -0.0243 \,]
\]

Valores reales:
\[
[\, 1.03,\ 3.821,\ 1.726,\ 0.934,\ 0.965 \,]
\]

Las predicciones iniciales difieren significativamente de los valores reales y se encuentran cercanas a cero. Esto es esperable, ya que el modelo aún no ha sido entrenado y los pesos fueron inicializados con valores aleatorios pequeños.
Por lo tanto, las predicciones iniciales **no son buenas**, pero establecen el punto de partida para el proceso de aprendizaje mediante descenso del gradiente [1].

**m) Diagrama del flujo de datos**
Respuesta: figures/m)Diagrama de flujo.jpeg
El diagrama muestra el flujo de datos del perceptrón: cada característica de entrada \(x_i\) se multiplica por su peso \(w_i\), los productos se suman en una combinación lineal \(w^T x\), y posteriormente se añade el sesgo \(b\) para obtener la salida \(y_{\text{pred}}\).

### 8. Función de pérdida

**n) Uso del cuadrado en la función de pérdida**
Las diferencias se elevan al cuadrado para penalizar con mayor severidad los errores grandes y asegurar que la función de pérdida sea suave y diferenciable. Esta propiedad es fundamental para aplicar el descenso del gradiente, ya que el MSE posee derivadas continuas respecto a los parámetros del modelo, tal como se describe en el Capítulo 2 del material del curso [1].

**o) Pérdida inicial del modelo**
La pérdida inicial del modelo, calculada con pesos aleatorios y sesgo inicial, es:

\[
\text{MSE}_{\text{inicial}} = 5.63
\]

Este valor representa el error del modelo antes del entrenamiento y se utilizará como referencia para evaluar la reducción de la pérdida tras aplicar el descenso del gradiente durante el proceso de aprendizaje.

**p) Ventajas del MSE en optimización**
El MSE es preferido frente al error absoluto medio (MAE) porque es diferenciable en todo su dominio. Esto permite calcular gradientes de forma estable y eficiente, facilitando la convergencia del algoritmo de descenso del gradiente, como se explica en el Capítulo 2 del material del curso [1].

**q) Estabilización de la función de pérdida**
Que la función de pérdida se estabilice significa que su valor deja de disminuir de forma significativa entre iteraciones. Esto indica que el modelo ha alcanzado un estado de convergencia, en el cual los pesos y el sesgo ya no cambian de manera apreciable y el algoritmo ha llegado a un mínimo de la función de pérdida [1].











## Referencias
[1] Material del curso. Capítulo 2: *Arquitectura de las redes neuronales*.  
[2] Scikit-learn documentation. *California Housing Dataset*.  
https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset  
[3] Scikit-learn documentation. *StandardScaler*.  
https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html



