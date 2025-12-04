# 📰 Proyecto Final: Detector de Noticias Falsas (Fake News Detector)

## Descripción del Proyecto

Detector de noticias falsas (REAL / FAKE) basado en Procesamiento de Lenguaje Natural (NLP) y Regresión Logística.

Incluye script de entrenamiento y una aplicación web de demostración con Streamlit.

------------------------------------------------------------------------

## Resultados y Métricas Clave

| Métrica | Valor |
| :--- | :--- |
| **Algoritmo Base** | Regresión Logística (Simbolista) |
| **Precisión (Accuracy)** | ≈ 98.5% |
| **Datos de Entrenamiento** | ≈ 44,000 Noticias |
| **Vectorización** | TF-IDF (Term Frequency - Inverse Document Frequency) |

------------------------------------------------------------------------

## Arquitectura y Proceso (Pipeline)

### 1. Preparación y Limpieza de Datos (Data Pipeline)

-   **Fuentes:** Fake and Real News (Kaggle). Datos cargados de `Fake.csv` y `True.csv` (aprox.
    44,000 documentos en total).
-   **Ingeniería de Características:** Se combinó el campo `title` y
    `text` para proporcionar al modelo un contexto semántico máximo.
-   **Limpieza (NLP):**
    -   Eliminación de *stopwords* y puntuación.
    -   **Anti-Sesgo Crítico:** Se eliminaron los metadatos de fuente
        (Ej: `WASHINGTON (REUTERS) -`) para asegurar que el modelo se
        enfoque en el **contenido** y no en la **fuente**.
-   **División:** Conjunto de entrenamiento (80%) y prueba (20%).

### 2. Vectorización (Traducción a Números)

-   **Herramienta:** `TfidfVectorizer`.
-   **Configuración Clave:** Se configuró para usar un
    `ngram_range=(1, 2)` (Unigramas y Bi-gramas) para capturar frases
    clave, y se limitó a las **5,000 features** más importantes
    (`max_features=5000`).

### 3. Modelo y Persistencia

-   **Modelo:** **Regresión Logística** por su velocidad, eficiencia e
    interpretabilidad.
-   **Persistencia:** Se guardó el modelo (`modelo_fake_news.pkl`) y el
    vectorizador (`vectorizer_tfidf.pkl`) con **joblib**.

------------------------------------------------------------------------

## Guía de Instalación y Ejecución

### 1. Clonar el Repositorio

``` bash
git clone https://github.com/MisaelCast/Proyecto-IA.git
cd Proyecto-IA
```

### 2. Configurar el Entorno Virtual (Venv)

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar Dependencias

``` bash
pip install pandas nltk scikit-learn joblib streamlit
```

### 4. Entrenar el Modelo

``` bash
python3 fake_news_ia.py
```

### 5. Ejecución Visual (Web App)

``` bash
streamlit run app.py
```

------------------------------------------------------------------------

## Autor

**Misael Castillo**

\[LinkedIn \]
