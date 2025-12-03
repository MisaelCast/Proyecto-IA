import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
import sys

# --- Configuración Inicial y Carga de Modelos ---
try:
    # Cargar los objetos entrenados (Modelo y Vectorizador)
    modelo = joblib.load('modelo_fake_news.pkl')
    vectorizer = joblib.load('vectorizer_tfidf.pkl')
    stop_words = set(stopwords.words("english"))
    
    # Mensaje de éxito en la consola (no en la app)
    print("Modelos y Vectorizador cargados exitosamente.")
except FileNotFoundError:
    st.error("Error: Archivos de modelo o vectorizador (.pkl) no encontrados.")
    st.error("Asegúrate de ejecutar 'fake_news_ia.py' primero para entrenar y guardar los modelos.")
    sys.exit()



# --- Función de Limpieza ---
def limpiar_texto(texto):
    # 1. ELIMINAR METADATA/FUENTE
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


# --- Lógica de la Aplicación Streamlit ---
st.title("📰 Detector de Noticias Falsas (IA)")
st.markdown("---")

st.header("Ingresa la noticia a clasificar:")

# Área de texto donde el usuario escribe la noticia
noticia_input = st.text_area(
    "Pega el texto de la noticia aquí:",
    height=200,
    placeholder="Ej: The European Union formally approved a new trade agreement with Canada on Thursday following a vote in the European Parliament in Brussels. Officials said the agreement is expected to strengthen economic cooperation...."
)

# Botón para activar la predicción
if st.button("Clasificar Noticia"):
    if noticia_input:
        with st.spinner('Clasificando...'):
            # 1. Limpiar el texto
            noticia_limpia = limpiar_texto(noticia_input)
            
            # 2. Vectorizar el texto (Transformar usando el vectorizador entrenado)
            noticia_vec = vectorizer.transform([noticia_limpia])
            
            # 3. Realizar la predicción
            prediccion = modelo.predict(noticia_vec)[0]
            
            # 4. Mostrar el resultado
            st.markdown("### Resultado de la Clasificación:")
            
            if prediccion == 'real':
                st.success(f"✅ La noticia es clasificada como **{prediccion.upper()}**")
                st.balloons() # Animación de celebración
            else:
                st.error(f"❌ La noticia es clasificada como **{prediccion.upper()}**")
                
    else:
        st.warning("Por favor, pega el texto de una noticia para clasificar.")