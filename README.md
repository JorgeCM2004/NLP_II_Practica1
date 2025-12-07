# NLP_II_Practica1: Clasificación de Género de Películas

Este proyecto implementa un sistema de clasificación de género de películas utilizando técnicas de Procesamiento del Lenguaje Natural (NLP). Se comparan modelos clásicos de Machine Learning con modelos basados en Transformers (DeBERTa) y se aplican técnicas de explicabilidad para entender las predicciones.

## Configuración del Entorno

Sigue estos pasos para configurar el entorno de ejecución. Puedes ejecutar estos comandos en tu terminal.

### 1. Crear un entorno virtual

```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate
```

```bash
# Unix/MacOS
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Instalar dependencias

Instala las librerías necesarias listadas en `requirements.txt`.

```bash
pip install -r requirements.txt
```

### 3. Instalar Kernel para Jupyter

Para ejecutar los notebooks dentro del entorno virtual:

```bash
pip install ipykernel
python -m ipykernel install --user --name=.venv --display-name "NLP_Practica1_Env"
```

## 📂 Estructura del Proyecto y Notebooks

El proyecto se divide en tres notebooks principales, cada uno con un propósito específico:

### 1. `F_Models_Training.ipynb` (Entrenamiento)

Este notebook es el punto de partida para **crear los modelos**.

- **Función**: Se encarga de descargar/cargar el dataset, preprocesar los textos y entrenar tanto los modelos básicos como el modelo Transformer.
- **Modelos Básicos**: Naive Bayes, Regresión Logística, SVM Lineal, Random Forest.
- **Transformer**: Realiza el fine-tuning de `deberta-v3-large`.
- **Salida**: Guarda los modelos entrenados en la carpeta `Models/`.

### 2. `F_Models_Tests.ipynb` (Evaluación)

Este notebook se utiliza para **evaluar el rendimiento** de los modelos ya entrenados.

- **Función**: Carga los modelos guardados desde el disco y evalúa su desempeño sobre el conjunto de test.
- **Métricas**: Genera reportes de clasificación (Accuracy, F1-Score) y matrices de confusión.
- **Uso**: Ejecuta este notebook si ya tienes los modelos en la carpeta `Models/` y quieres ver resultados sin re-entrenar. Aun asi, si no estan entrenados y lo quieres ejecutar, este los entrenara primero y luego analiza los resultados.

### 3. `F_Explicabilidad.ipynb` (Explicabilidad)

Este notebook aplica técnicas de **Inteligencia Artificial Explicable (XAI)** para interpretar las predicciones.

- **Modelos Clásicos**: Utiliza **LIME** para visualizar qué palabras contribuyen positiva o negativamente a la clasificación en modelos como Regresión Logística.
- **Transformer**: Utiliza **Integrated Gradients** (vía librería `Captum`) para visualizar la atribución de importancia token a token en el modelo DeBERTa.
- **Objetivo**: Ayudar a entender "por qué" el modelo clasificó una película en un género específico.
