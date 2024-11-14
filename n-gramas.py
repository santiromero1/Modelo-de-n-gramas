import re
from nltk.tokenize import word_tokenize
import nltk

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

print(texto_limpio[:500])  # Imprime los primeros 500 caracteres del texto limpio