# Examen Práctico – Redes Neuronales

## Henry Santiago Cifuentes Serrano
## Juan Diego Cifuentes oliva

## Parte I: Exploración del Conjunto de Datos

### 1. Exploración inicial del dataset

**a) Número de observaciones (n):**
El conjunto de datos California Housing contiene **20640 observaciones**.

**b) Número de características por observación:**
Cada observación del dataset está descrita por **8 características numéricas.**

**c) Vector de características de una observación:**
Siguiendo la convención introducida en el **Capítulo 2 del material del curso (Arquitectura de las redes neuronales)**, las características de una observación se representan mediante un vector columna \(\mathbf{x}\), cuyos componentes corresponden a cada una de las variables de entrada. En este caso, al tratarse de 8 características, el vector se escribe como:

$$\mathbf{x} = \begin{pmatrix} x_1 \\ x_2 \\ x_3 \\ x_4 \\ x_5 \\ x_6 \\ x_7 \\ x_8 \end{pmatrix}$$

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


## Parte 3: Retropropagación y gradiente descendente

### 9. Cálculo gradiente

**r) Derivación de \(\partial L / \partial w\)**
Partimos de la función de pérdida para regresión:

\[
L = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i)^2
\]

donde, para una activación lineal, las predicciones del modelo están dadas por:

\[
\hat{y} = Xw + b
\]

Definimos el vector de error como:

\[
e = \hat{y} - y
\]

La función de pérdida puede escribirse en forma matricial como:

\[
L = \frac{1}{m} e^T e
\]

Al derivar la pérdida respecto al vector de pesos \(w\) y aplicar las reglas de derivación matricial, se obtiene:

\[
\frac{\partial L}{\partial w} = \frac{2}{m} X^T (\hat{y} - y)
\]

Esta expresión indica cómo varía la pérdida ante cambios en los pesos del modelo y corresponde a la formulación presentada en el **Capítulo 2 del material del curso** para problemas de regresión entrenados mediante descenso del gradiente \[1].

**s) Derivación de \(\partial L / \partial b\)**
El sesgo \(b\) afecta de igual forma a todas las predicciones del modelo, ya que:

\[
\hat{y}_i = w^T x_i + b
\]

Al derivar la función de pérdida respecto al sesgo, se obtiene:

\[
\frac{\partial L}{\partial b} = \frac{2}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i)
\]

Este gradiente representa el error promedio acumulado del modelo y determina cómo debe ajustarse el sesgo durante el proceso de entrenamiento \[1].

**t) Interpretación geométrica del gradiente**
El gradiente de una función señala la **dirección de máxima pendiente ascendente** de la superficie que representa dicha función.  
En la metáfora de la montaña descrita en el **Capítulo 2**, la función de pérdida corresponde a la altura del terreno y los pesos del modelo representan la posición del excursionista.
Moverse en la dirección del gradiente implica ascender más rápidamente; por ello, para minimizar la pérdida, el algoritmo de aprendizaje avanza en la dirección opuesta al gradiente.

**u) Significado del signo negativo en la regla de actualización**
La regla de actualización de los pesos está dada por:

\[
w \leftarrow w - \eta \nabla L
\]

El signo negativo indica que el objetivo del aprendizaje es **minimizar** la función de pérdida. Dado que el gradiente apunta hacia el aumento más rápido de la pérdida, restarlo permite desplazarse en la dirección contraria, conduciendo al modelo hacia un mínimo de la función. Este principio constituye la base del algoritmo de **descenso del gradiente**, tal como se presenta en el material del curso \[1].

### 10. Cálculo de gradientes

**v) Implementación de los gradientes**
La función `calcular_gradientes` se implementó utilizando las expresiones derivadas en el punto anterior. El gradiente respecto a los pesos se calcula como:

\[
\frac{\partial L}{\partial w} = \frac{2}{m} X^T (\hat{y} - y)
\]

mientras que el gradiente respecto al sesgo está dado por:

\[
\frac{\partial L}{\partial b} = \frac{2}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i)
\]

Estas fórmulas permiten cuantificar cómo deben ajustarse los parámetros del modelo para reducir la pérdida, de acuerdo con el algoritmo de descenso del gradiente descrito en el **Capítulo 2 del material del curso** \[1].

**w) Forma del gradiente respecto a los pesos**
El gradiente \(dw\) tiene la misma forma que el vector de pesos \(w\), es decir, una matriz de dimensión \((n, 1)\). Esto es necesario porque la actualización de los parámetros se realiza mediante la operación:

\[
w \leftarrow w - \eta \, dw
\]

La coherencia dimensional garantiza que la resta esté bien definida y que cada peso se actualice con su gradiente correspondiente.

**x) Gradientes cuando las predicciones son perfectas**
Si todas las predicciones del modelo coinciden exactamente con los valores reales, se cumple que:

\[
\hat{y} = y
\]

En este caso, el error es nulo y tanto \(dw\) como \(db\) toman el valor cero. Esto indica que el modelo ha alcanzado un mínimo de la función de pérdida y que no se requieren más actualizaciones de los parámetros, lo cual corresponde al concepto de convergencia \[1].

### 11. Actualización de parámetros

**y) Regla de actualización del gradiente descendente**
La actualización de los parámetros del perceptrón se realiza mediante el descenso del gradiente, siguiendo las expresiones:

\[
\mathbf{w} \leftarrow \mathbf{w} - \eta \nabla_{\mathbf{w}} L
\quad \text{y} \quad
b \leftarrow b - \eta \frac{\partial L}{\partial b}
\]

donde \(\eta\) es la tasa de aprendizaje y \(\nabla L\) indica el gradiente de la función de pérdida respecto a los parámetros del modelo [1].

**z) Tasa de aprendizaje (\(\eta\))**
La tasa de aprendizaje controla el tamaño del paso que se da en cada iteración del descenso del gradiente. Si \(\eta\) es **demasiado grande**, el algoritmo puede oscilar o divergir y no converger al mínimo. Si \(\eta\) es **demasiado pequeña**, la convergencia será muy lenta, aumentando el tiempo de entrenamiento. Por ello, la elección de \(\eta\) es crítica para un aprendizaje estable y eficiente [1].

**aa) Ejecución de una iteración completa**
Se ejecutó una iteración completa del algoritmo: propagación hacia adelante, cálculo de la pérdida, cálculo de gradientes y actualización de parámetros. La pérdida **disminuyó** tras la actualización, pasando de **92964.53** a **90167.59**, lo cual indica que el perceptrón ajustó sus parámetros en la dirección que reduce el error. Este comportamiento confirma que el algoritmo de retropropagación y el descenso del gradiente están funcionando correctamente y que el modelo comienza a aprender a partir de los datos [1].


## Parte 4: Entrenamiento completo

### 12.	Implementación del ciclo de entrenamiento completo

**bb) Implementación del ciclo de entrenamiento**
La función `entrenar_perceptron` se completó integrando todas las etapas del aprendizaje supervisado vistas anteriormente: propagación hacia adelante, cálculo de la pérdida mediante MSE, cálculo de los gradientes y actualización de los parámetros usando descenso del gradiente.
En cada época se almacena el valor de la pérdida para analizar la evolución del entrenamiento, siguiendo el flujo descrito en el Capítulo 2 del material del curso [1].

**cc) Entrenamiento del modelo y gráfica de la pérdida**
El modelo se entrenó utilizando una tasa de aprendizaje `learning_rate = 0.01` y `epochs = 1000`. Se graficó el historial de la función de pérdida (MSE) en función del número de épocas, observándose una disminución pronunciada al inicio del entrenamiento y una posterior estabilización.
La gráfica de la evolución de la pérdida se incluye como parte de la entrega en figures/cc)curva_perdida_entrenamiento.png.

**dd) Convergencia del modelo**
Sí, el modelo converge. Esto se evidencia en la gráfica de la pérdida, donde el MSE disminuye rápidamente en las primeras épocas y luego se estabiliza alrededor de un valor casi constante. La ausencia de oscilaciones grandes o crecimiento descontrolado de la pérdida indica que el algoritmo de descenso del gradiente alcanzó una región cercana a un mínimo, lo que corresponde al concepto de convergencia descrito en el Capítulo 2 [1].

**ee) Comparación de diferentes tasas de aprendizaje**
Respuesta imagenes: ee)Comparación_de_tasas_de_aprendizaje01 ; ee)Comparación_de_tasas_de_aprendizaje02
Se entrenó el perceptrón utilizando cuatro tasas de aprendizaje:  
`η = 0.001`, `0.01`, `0.1` y `1.0`, y se comparó la evolución de la pérdida (MSE) a lo largo de las épocas.
Para `η = 0.001`, la pérdida disminuye de forma estable pero muy lenta, indicando un aprendizaje seguro aunque poco eficiente.  
Con `η = 0.01`, el modelo presenta una caída rápida de la pérdida y una posterior estabilización, logrando el mejor equilibrio entre velocidad de convergencia y estabilidad.  
En el caso de `η = 0.1`, la pérdida disminuye bruscamente al inicio pero luego oscila alrededor de un valor mayor, lo que sugiere que el paso de actualización es demasiado grande y sobrepasa el mínimo.  
Finalmente, con `η = 1.0`, la pérdida diverge rápidamente y aparecen valores infinitos y `NaN`, evidenciando inestabilidad numérica y fallo del descenso del gradiente.
En conclusión, la tasa de aprendizaje `η = 0.01` resulta la más adecuada para este problema, mientras que `η = 1.0` provoca divergencia, tal como se anticipa teóricamente en el estudio del descenso del gradiente presentado en el Capítulo 2 del material del curso [1].

### 13.	Evaluación en el conjunto de prueba

**ff) Predicciones en el conjunto de prueba**
Utilizando los parámetros entrenados \(w\) y \(b\), se calcularon las predicciones sobre el conjunto de prueba normalizado mediante la propagación hacia adelante:
\[
\hat{y} = X_{\text{test}} w + b
\]
El vector de predicciones obtenido tiene forma \((4128, 1)\), coherente con el número de observaciones del conjunto de prueba.

**gg) Error cuadrático medio (MSE) en test**
El error cuadrático medio en el conjunto de prueba fue:
\[
\text{MSE}_{\text{test}} \approx 8617.35
\]
Este valor es **menor** que el MSE final del entrenamiento (\(\approx 34719.78\)), lo cual indica que el modelo **generaliza adecuadamente** y no presenta sobreajuste. La normalización de las características contribuye a esta buena capacidad de generalización.

**hh) Scatter plot: valores reales vs predicciones**
Respuesta imagen: figures/hh)Predicciones_vs_valores_reales.png
Se construyó un gráfico de dispersión comparando los valores reales \(y_{\text{test}}\) con las predicciones \(\hat{y}\), junto con la recta ideal \(y = x\).
Los puntos se distribuyen razonablemente cerca de la diagonal, mostrando una correlación clara entre valores reales y predichos, aunque con dispersión, esperable en un modelo lineal sencillo.

**ii) Coeficiente de determinación \(R^2\)**
El coeficiente de determinación obtenido fue:
\[
R^2 \approx 0.577
\]
Este valor indica que el modelo explica aproximadamente el **57.7% de la varianza** de la variable objetivo en el conjunto de prueba. Para un perceptrón lineal sin capas ocultas, este resultado es consistente y confirma que el modelo captura una parte significativa de la estructura del problema.


## Reflexión final: Limitaciones del perceptrón simple

### 14. Limitaciones del perceptrón simple

**jj) Tipo de relación que puede modelar un perceptrón simple**
Un perceptrón simple (sin capas ocultas) solo puede modelar **relaciones lineales** entre las características de entrada y la variable objetivo.  
En este caso, el modelo aprende una combinación lineal de las variables del conjunto de datos para aproximar el precio de las viviendas, tal como se describe en el **Capítulo 2 del material del curso** [1].

**kk) Extensión del modelo para relaciones no lineales**
Si la relación real entre las características y el precio de las viviendas fuera altamente no lineal, el modelo podría extenderse incorporando **capas ocultas con funciones de activación no lineales**, dando lugar a un **perceptrón multicapa (MLP)**.  
Estas capas permiten transformar el espacio de características original y capturar patrones complejos que no pueden representarse mediante una sola combinación lineal [1].

**ll) Teorema de Aproximación Universal y uso de redes profundas**
El **Teorema de Aproximación Universal** garantiza que una red neuronal con **una sola capa oculta**, un número suficiente de neuronas y una función de activación no lineal puede aproximar cualquier función continua sobre un dominio compacto con la precisión deseada [1].  
Sin embargo, en la práctica se utilizan **redes profundas** porque permiten representar funciones complejas de forma más eficiente, usando menos neuronas por capa, mejor generalización y una estructura jerárquica que facilita el aprendizaje de patrones complejos.


## Referencias
[1] Material del curso. Capítulo 2: *Arquitectura de las redes neuronales*.  
[2] Scikit-learn documentation. *California Housing Dataset*.  
https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset  
[3] Scikit-learn documentation. *StandardScaler*.  
https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html



