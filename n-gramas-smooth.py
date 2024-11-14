import re
from nltk import ngrams
from nltk.tokenize import word_tokenize
from collections import defaultdict
import nltk
import random

# 1. Conseguir y limpiar un corpus de textos
nltk.download('punkt')

# Cargar el archivo
with open('Gatsby.txt', 'r', encoding='utf-8') as file:
    texto = file.read()

# Limpiar el texto
inicio_contenido = texto.find("In my younger and more vulnerable years")
fin_contenido = texto.find("*** END OF THIS PROJECT GUTENBERG EBOOK THE GREAT GATSBY ***")
texto = texto[inicio_contenido:fin_contenido]  # Elimina metadatos
texto = texto.lower()  # Convierte a minúsculas
texto = re.sub(r'[^\w\s]', '', texto)  # Elimina puntuación

# Tokenización sin eliminación de stopwords
tokens = word_tokenize(texto)

# Juntar en texto limpio sin stopwords
texto_limpio = ' '.join(tokens)

# 2. Modelo de n-gramas con suavizado de Laplace

# Función de construcción con suavizado de Laplace
def construir_ngramas(tokens, n, suavizado=True):
    ngramas = defaultdict(int)  # Usamos defaultdict para contar la frecuencia de cada n-grama
    vocabulario = set(tokens)  # Conjunto de palabras únicas (vocabulario)
    total_tokens = len(tokens)  # Total de tokens en el corpus
    
    for ngrama in ngrams(tokens, n):  # Generamos los n-gramas
        ngramas[ngrama] += 1  # Incrementamos la frecuencia del n-grama
    
    # Si se aplica suavizado de Laplace
    if suavizado:
        # Suavizado de Laplace: Se le agrega 1 a cada frecuencia y se ajusta la probabilidad
        for ngrama in ngramas:
            ngramas[ngrama] += 1
        
        # Número total de n-gramas posibles
        total_ngramas_posibles = len(vocabulario) ** (n - 1)
        
        # Aplicamos el suavizado y normalizamos las probabilidades
        for ngrama in ngramas:
            ngramas[ngrama] /= (total_tokens + total_ngramas_posibles)
    
    return ngramas

# Construir n-gramas con suavizado
n = 3  # Probar con trigramas (n=3) o ajusta según lo necesites
ngramas = construir_ngramas(tokens, n)

# Mostrar los primeros 10 n-gramas y sus frecuencias
print("Primeros 10 n-gramas:")
print(list(ngramas.items())[:10])

# Función de generación con suavizado
def generar_texto(modelo_ngramas, n, longitud, inicio):
    texto_generado = inicio.split()  # Comenzamos con el inicio dado
    for _ in range(longitud):
        contexto = tuple(texto_generado[-(n-1):])  # Tomamos las últimas n-1 palabras
        opciones = {ngramas: frecuencia for ngramas, frecuencia in modelo_ngramas.items() if ngramas[:-1] == contexto}
        
        if not opciones:
            break  # Si no hay opciones posibles, terminamos la generación

        # Elegir la palabra siguiente de acuerdo a la frecuencia
        siguiente_palabra = random.choices(list(opciones.keys()), list(opciones.values()))[0][-1]
        texto_generado.append(siguiente_palabra)
    
    return ' '.join(texto_generado)

# Generar un texto de ejemplo (con trigramas, longitud 50 palabras)
inicio = "in my younger and more"
texto_generado = generar_texto(ngramas, n, 50, inicio)
print("\nTexto generado:")
print(texto_generado)
