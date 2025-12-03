import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import re 
import joblib 
import sys

# NOTA: Asegúrate de que las stopwords de NLTK estén descargadas:
# python3 -c 'import nltk; nltk.download("stopwords"); nltk.download("punkt")'

# --- PASO 2. Cargar y Unir los Datos ---
print("--- 1. Carga de Datos ---")
try:
    # Cargar datasets, seleccionando 'title', 'text' y 'label'
    fake = pd.read_csv("Fake.csv")
    true = pd.read_csv("True.csv")

    # Agregar etiquetas
    fake["label"] = "fake"
    true["label"] = "real"

    # Unir datasets (manteniendo solo las columnas necesarias)
    df = pd.concat([fake[['title', 'text', 'label']], true[['title', 'text', 'label']]], ignore_index=True)

    # *** SOLUCIÓN CLAVE: Combinar Título y Texto para más contexto ***
    df['full_text'] = df['title'].astype(str) + ' ' + df['text'].astype(str)

    # Revisar y limpiar valores nulos
    print(f"Total de noticias antes de limpieza de nulos: {len(df)}")
    df.dropna(subset=['full_text', 'label'], inplace=True) 
    df.fillna('', inplace=True) 

    print(f"Total de noticias después de limpieza de nulos: {len(df)}")
    print("Distribución de etiquetas:")
    print(df['label'].value_counts())
    print("-" * 30)

except FileNotFoundError:
    print("Error: Asegúrate de que los archivos 'Fake.csv' y 'True.csv' estén en la misma carpeta.")
    sys.exit()

# --- PASO 3. Limpiar y Preparar el Texto (NLP) ---
print("--- 2. Preprocesamiento (NLP) ---")
try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    print("Error: Necesitas descargar las stopwords de NLTK. Ejecuta: python3 -c 'import nltk; nltk.download(\"stopwords\")'")
    sys.exit()

def limpiar_texto(texto):
    # 1. ELIMINAR METADATA/FUENTE: Quita patrones como 'LUGAR (AGENCIA) - ' (ej. WASHINGTON (REUTERS) - )
    texto = re.sub(r'([A-Z\s]+)\s*\((REUTERS|AP|AFP)\)\s*\-\s*', '', str(texto), flags=re.IGNORECASE)
    
    # 2. Convertir a minúsculas
    texto = str(texto).lower()
    
    # 3. Eliminar puntuación, números y caracteres especiales
    texto = re.sub(r'[^a-z\s]', '', texto) 
    
    # 4. Tokenización con split() 
    tokens = texto.split() 

    # 5. Filtrar stopwords y tokens de una sola letra
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
    return " ".join(tokens)

# Aplicar limpieza a la columna que combina título y texto
df["clean_text"] = df["full_text"].apply(limpiar_texto)
print(df[["title", "text", "clean_text", "label"]].head())
print("-" * 30)

# --- PASO 4. Convertir texto en números (TF-IDF) y 5. Dividir datos ---
print("--- 3. Vectorización y División de Datos ---")
X = df["clean_text"] # Usar la columna limpia que combina título y texto
y = df["label"]

# Vectorización TF-IDF OPTIMIZADA: Rápida y robusta (Bi-gramas, 5000 features)
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2)) 
X_tfidf = vectorizer.fit_transform(X)

# Dividir datos (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42
)
print(f"Tamaño del set de entrenamiento: {X_train.shape[0]} ({X_train.shape[1]} features)")
print(f"Tamaño del set de prueba: {X_test.shape[0]}")
print("-" * 30)

# --- PASO 6. Entrenar el modelo (Regresión Logística) ---
print("--- 4. Entrenamiento del Modelo ---")
modelo = LogisticRegression(max_iter=1000, solver='liblinear', random_state=42)
modelo.fit(X_train, y_train)
print("¡Modelo entrenado exitosamente! ✅")

# *** GUARDAR EL MODELO Y EL VECTORIZADOR ***
joblib.dump(modelo, 'modelo_fake_news.pkl')
joblib.dump(vectorizer, 'vectorizer_tfidf.pkl')
print("Modelos guardados exitosamente como 'modelo_fake_news.pkl' y 'vectorizer_tfidf.pkl'")
print("-" * 30)

# --- PASO 7. Evaluar el modelo ---
print("--- 5. Evaluación del Modelo ---")
y_pred = modelo.predict(X_test)

print("Accuracy (Precisión General):", accuracy_score(y_test, y_pred))
print("\nReporte de Clasificación:\n", classification_report(y_test, y_pred))
print("-" * 30)

# --- PASO 8. Probar con noticias nuevas ---
print("--- 6. Prueba con Noticias Nuevas ---")

# Lista de noticias a probar
noticias_nuevas = [
    # 1. Caso financiero/político formal (Debería salir REAL)
    "The Federal Reserve announced on Wednesday that it will maintain the benchmark interest rate within the current range of 5.25% to 5.50%, citing steady economic growth and easing inflation. Federal Reserve Chair Jerome Powell stated during a press briefing in Washington that future rate decisions will depend on labor market data and inflation trends over the coming months.",
    # 2. Caso claramente FAKE/Conspiración
    "A secret meeting was held at the UN headquarters where delegates voted to replace all sugary drinks with green juice to boost the global population by 500 years.",
    # 3. Caso político REAL
    "President Joe Biden announced a new infrastructure plan, stating, 'This investment will create millions of jobs across the country.'"
]

# Aplicar la misma limpieza y vectorización a la lista
noticias_limpias = [limpiar_texto(n) for n in noticias_nuevas]
noticias_vec = vectorizer.transform(noticias_limpias)

# Realizar las predicciones
predicciones = modelo.predict(noticias_vec)

# Mostrar resultados
for i, (noticia, prediccion) in enumerate(zip(noticias_nuevas, predicciones)):
    print(f"\nNoticia {i+1} (Inicio): {noticia[:50]}...")
    print(f"Predicción: {prediccion.upper()}")

print("--- FIN DEL PROYECTO ---")