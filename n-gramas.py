## Trabajo Práctico 3 - Lenguaje ##

import re
from nltk import ngrams
from nltk.tokenize import word_tokenize
from collections import defaultdict
import nltk
import random

# 1. Conseguir y limpiar un corpus de textos

# Descargar datos necesarios de NLTK
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

## print(texto_limpio[:500])  # Imprime los primeros 500 caracteres del texto limpio

##########################################################################################################################################

# 2. Modelo de n-gramas


# Función de construcción
def construir_ngramas(tokens, n):
    ngramas = defaultdict(int)  # Usamos defaultdict para contar la frecuencia de cada n-grama
    for ngrama in ngrams(tokens, n):  # Generamos los n-gramas
        ngramas[ngrama] += 1  # Incrementamos la frecuencia del n-grama
    return ngramas

# Construir n-gramas para diferentes valores de n
n = 5  # Ejemplo con trigramas (n=3)
ngramas = construir_ngramas(tokens, n)

# Mostrar los primeros 10 n-gramas y sus frecuencias
print("Primeros 10 n-gramas:")
print(list(ngramas.items())[:10])


# Función de generación
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