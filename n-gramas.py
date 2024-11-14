## Modelo de n-gramas ##

import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
# Descargar datos necesarios de NLTK
nltk.download('stopwords')
nltk.download('punkt')


# Cargar el archivo
with open('nombre_del_libro.txt', 'r', encoding='utf-8') as file:
    texto = file.read()

# Limpiar el texto
inicio_contenido = texto.find("Inicio del contenido")  # Ajusta esto según el archivo
fin_contenido = texto.find("Fin del contenido")
texto = texto[inicio_contenido:fin_contenido]  # Elimina metadatos
texto = texto.lower()  # Convierte a minúsculas
texto = re.sub(r'[^\w\s]', '', texto)  # Elimina puntuación

# Tokenización y eliminación de stopwords
tokens = word_tokenize(texto)
stop_words = set(stopwords.words('spanish'))
tokens = [word for word in tokens if word not in stop_words]

# Juntar en texto limpio
texto_limpio = ' '.join(tokens)

print(texto_limpio[:500])  # Imprime los primeros 500 caracteres del texto limpio
