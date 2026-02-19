# Examen Práctico – Redes Neuronales

## Henry Santiago Cifuentes Serrano
## Juan Diego Cifuentes Oliva

Este repositorio contiene la solución al **Examen Práctico de Redes Neuronales**, cuyo objetivo es implementar desde cero un **perceptrón para regresión**, aplicándolo al conjunto de datos **California Housing**.

El proyecto integra tanto el desarrollo práctico en Python como el análisis teórico solicitado en el examen, siguiendo los conceptos presentados en el **Capítulo 1 y 2 del material del curso (Arquitectura de las redes neuronales)**.

## Estructura del proyecto

california-housing-perceptron/
│
├── notebooks/
│   └── exploration.ipynb
│       → Notebook principal con todo el código del proyecto:
│         carga de datos, exploración, implementación del perceptrón,
│         entrenamiento, evaluación y generación de gráficas.
│
├── report/
│   └── respuestas_teoricas.md
│       → Documento con todas las respuestas teóricas del examen,
│         debidamente argumentadas y referenciadas.
│
├── figures/
│   ├── cc) curva_perdida_entrenamiento.png
│   ├── ee) Comparación_de_tasas_de_aprendizaje01.png
│   ├── ee) Comparación_de_tasas_de_aprendizaje02.png
│   └── hh) Predicciones_vs_valores_reales.png
│       → Gráficas generadas durante el entrenamiento y la evaluación
│         del modelo, solicitadas como parte del entregable.
│
├── src/
│   → Carpeta reservada para una posible modularización futura del código.
│     En esta entrega, el desarrollo se realiza íntegramente en el notebook.
│
├── requirements.txt
│   → Lista de dependencias necesarias para ejecutar el proyecto.
│
├── .gitignore
│   → Archivos y carpetas excluidos del control de versiones (por ejemplo, venv).
│
├── venv/
│   → Entorno virtual de Python utilizado durante el desarrollo (no versionado).
│
└── README.md
│   → Descripción general del proyecto y guía de uso.


## Contenido del proyecto

El desarrollo del examen se divide en las siguientes partes:

- **Parte I – Exploración del conjunto de datos**  
  Análisis del dataset California Housing, identificación de variables,
  tipo de problema y preprocesamiento mediante normalización.

- **Parte II – Arquitectura del perceptrón**  
  Implementación de la propagación hacia adelante, suma ponderada y función
  de pérdida (MSE), conectando cada paso con la teoría.

- **Parte III – Retropropagación y descenso del gradiente**  
  Derivación matemática de los gradientes e implementación del algoritmo
  de aprendizaje.

- **Parte IV – Entrenamiento completo y evaluación**  
  Entrenamiento del modelo, análisis de convergencia, comparación de tasas
  de aprendizaje y evaluación sobre el conjunto de prueba.

- **Reflexión final**  
  Discusión sobre las limitaciones del perceptrón simple y su relación con
  modelos más complejos como el perceptrón multicapa.

## Resultados principales

- El modelo converge correctamente para una tasa de aprendizaje
  `learning_rate = 0.01`.
- Tasas de aprendizaje altas (por ejemplo, `learning_rate = 1.0`) producen
  divergencia del algoritmo.
- Evaluación en el conjunto de prueba:
  - **MSE (test):** ≈ 8617
  - **Coeficiente R²:** ≈ 0.58

Estos resultados son coherentes con las limitaciones de un modelo lineal
entrenado mediante descenso del gradiente.

## Requisitos

Las dependencias necesarias para ejecutar el proyecto se encuentran en
`requirements.txt`.

Instalación:

```bash
pip install -r requirements.txt


