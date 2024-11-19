import re
import nltk
from nltk.tokenize import word_tokenize

# Descargar datos necesarios de NLTK (solo la primera vez)
nltk.download('punkt', quiet=True)

class TextProcessor:
    """
    Clase para cargar, limpiar y tokenizar un corpus de texto.
    """
    def __init__(self, filepath, start_marker, end_marker):
        """
        Inicializa el procesador de texto.

        Args:
            filepath (str): Ruta al archivo de texto.
            start_marker (str): Subcadena que indica el inicio del contenido relevante.
            end_marker (str): Subcadena que indica el fin del contenido relevante.
        """
        self.filepath = filepath
        self.start_marker = start_marker
        self.end_marker = end_marker
        self.tokens = []

    def load_and_clean_text(self):
        """
        Carga el archivo de texto, elimina metadatos, convierte a minúsculas y elimina puntuación.
        """
        try:
            with open(self.filepath, 'r', encoding='utf-8') as file:
                texto = file.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"El archivo {self.filepath} no se encontró.")

        # Encontrar los índices de inicio y fin del contenido relevante
        inicio_contenido = texto.find(self.start_marker)
        # print(f"Start marker '{self.start_marker}' found at index: {inicio_contenido}")
        fin_contenido = texto.find(self.end_marker)
        # print(f"End marker '{self.end_marker}' found at index: {fin_contenido}")

        if inicio_contenido == -1 or fin_contenido == -1:
            raise ValueError("No se encontraron los marcadores de inicio o fin en el texto.")

        # Extraer el contenido relevante usando los índices
        texto = texto[inicio_contenido:fin_contenido]

        # Convertir a minúsculas
        texto = texto.lower()

        # Eliminar puntuación usando expresiones regulares
        texto = re.sub(r'[^\w\s]', '', texto)

        # Tokenizar el texto
        self.tokens = word_tokenize(texto)
        
    def get_tokens(self):
        """
        Retorna la lista de tokens del texto procesado.

        Returns:
            list: Lista de tokens.
        """
        return self.tokens
    def get_text(self):
        try:
            with open(self.filepath, 'r', encoding='utf-8') as file:
                texto = file.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"El archivo {self.filepath} no se encontró.")

        # Encontrar los índices de inicio y fin del contenido relevante
        inicio_contenido = texto.find(self.start_marker)
        print(f"Start marker '{self.start_marker}' found at index: {inicio_contenido}")
        fin_contenido = texto.find(self.end_marker)
        print(f"End marker '{self.end_marker}' found at index: {fin_contenido}")

        if inicio_contenido == -1 or fin_contenido == -1:
            raise ValueError("No se encontraron los marcadores de inicio o fin en el texto.")

        # Extraer el contenido relevante usando los índices
        texto = texto[inicio_contenido:fin_contenido]

        # Convertir a minúsculas
        texto = texto.lower()
        return texto
    
    def get_training_segments(self, segment_length=150):
        """Divide el texto de entrenamiento en segmentos de longitud especificada."""
        segments = []
        tokens = self.tokens
        for i in range(0, len(tokens), segment_length):
            segment = tokens[i:i + segment_length]
            if len(segment) == segment_length:
                segments.append(" ".join(segment))
        return segments

def main():
    # 1. Conseguir y limpiar un corpus de textos
    filepath = 'Gatsby.txt'
    start_marker = "In my younger and more vulnerable years"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK THE GREAT GATSBY ***"

    text_processor = TextProcessor(filepath, start_marker, end_marker)
    text_processor.load_and_clean_text()
    tokens = text_processor.get_tokens()
    print(f"Cantidad de tokens: {len(tokens)}")
    print("\nPrimeros 500 caracteres del texto limpio:")
    print(' '.join(tokens)[:500])

