import joblib
import re
from nltk.corpus import stopwords
import pandas as pd
import sys

# --- Carga de Objetos Entrenados ---
try:
    modelo = joblib.load('modelo_fake_news.pkl')
    vectorizer = joblib.load('vectorizer_tfidf.pkl')
    
    # Intentar cargar stopwords (si fallara, el programa termina con un error)
    try:
        stop_words = set(stopwords.words("english"))
    except LookupError:
        print("Error: Necesitas descargar las stopwords de NLTK.")
        print("Ejecuta: python3 -c 'import nltk; nltk.download(\"stopwords\")'")
        sys.exit()

    print("Modelos cargados exitosamente. Listo para clasificar. ✅")
except FileNotFoundError:
    print("Error: Los archivos 'modelo_fake_news.pkl' o 'vectorizer_tfidf.pkl' no se encontraron.")
    print("Asegúrate de ejecutar 'fake_news_ia.py' primero para entrenar y guardar los modelos.")
    sys.exit()

# --- Función de Limpieza (Debe ser IDÉNTICA a la usada en el entrenamiento) ---
def limpiar_texto(texto):
    # 1. ELIMINAR METADATA/FUENTE (ajuste para la nueva version que combina titulo y texto)
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


# --- PRUEBA RÁPIDA DE CLASIFICACIÓN ---
noticias_nuevas = [
    # Noticias de prueba para clasificar
    "The Federal Reserve announced on Wednesday that it will maintain the benchmark interest rate within the current range of 5.25% to 5.50%, citing steady economic growth and easing inflation. Federal Reserve Chair Jerome Powell stated during a press briefing in Washington that future rate decisions will depend on labor market data and inflation trends over the coming months.",
    "A secret meeting was held at the UN headquarters where delegates voted to replace all sugary drinks with green juice to boost the global population by 500 years.",
    "President Joe Biden announced a new infrastructure plan, stating, 'This investment will create millions of jobs across the country.'",
    "The European Union formally approved a new trade agreement with Canada on Thursday following a vote in the European Parliament in Brussels. Officials said the agreement is expected to strengthen economic cooperation and reduce tariffs on industrial goods over the next five years.",
    "The World Health Organization reported on Monday that global vaccination rates have increased by 12 percent compared to last year, according to data collected from member states. WHO Director-General Tedros Adhanom Ghebreyesus emphasized the importance of continued international cooperation to prevent future outbreaks.",
    "Apple Inc. unveiled its latest software update during a developer conference in California on Tuesday. The update introduces enhanced security features, improved battery management, and performance optimizations for supported devices. The company stated that the update will be available to the public next month.",
    "For the first time, scientists are tracking the migration of monarch butterflies across much of North America, actively monitoring individual insects on journeys from as far away as Ontario all the way to their overwintering colonies in central Mexico.This long-sought achievement could provide crucial insights into the poorly understood life cycles of hundreds of species of butterflies, bees and other flying insects at a time when many are in steep decline."
]

print("\n--- Clasificación de Noticias Nuevas (Rápida) ---")

# Aplicar la misma limpieza y vectorización (transformar, no fit)
noticias_limpias = [limpiar_texto(n) for n in noticias_nuevas]
noticias_vec = vectorizer.transform(noticias_limpias)

# Realizar las predicciones
predicciones = modelo.predict(noticias_vec)

# Mostrar resultados
for i, (noticia, prediccion) in enumerate(zip(noticias_nuevas, predicciones)):
    print(f"\nNoticia {i+1} (Inicio): {noticia[:50]}...")
    print(f"Predicción: {prediccion.upper()}")

print("--- FIN DEL PROYECTO ---")