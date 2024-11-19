import numpy as np
import pandas as pd
import nltk
from TextProcessor import TextProcessor
# Descargar datos necesarios de NLTK (solo la primera vez)
nltk.download('punkt', quiet=True)

class CooccurrenceModel:
    """
    Clase para construir un modelo de coocurrencias y generar texto basado en coocurrencias.
    """
    def __init__(self, window_size=2):
        """
        Inicializa el modelo de coocurrencias.

        Args:
            window_size (int): Tamaño de la ventana de contexto para contar coocurrencias.
        """
        self.window_size = window_size
        self.vocab = []
        self.word_to_index = {}
        self.cooccurrence_matrix = None

    def build_cooccurrence_matrix(self, tokens):
        """
        Construye la matriz de coocurrencias a partir de una lista de tokens.

        Args:
            tokens (list): Lista de tokens del texto.
        """
        self.vocab = sorted(set(tokens))
        self.word_to_index = {word: i for i, word in enumerate(self.vocab)}
        vocab_size = len(self.vocab)
        self.cooccurrence_matrix = np.zeros((vocab_size, vocab_size), dtype=int)

        for index, word in enumerate(tokens):
            word_idx = self.word_to_index[word]
            start = max(0, index - self.window_size)
            end = min(len(tokens), index + self.window_size + 1)

            for i in range(start, end):
                if i != index:  # Evitar la palabra misma
                    neighbor_idx = self.word_to_index[tokens[i]]
                    self.cooccurrence_matrix[word_idx][neighbor_idx] += 1

        # Convertimos la matriz en un DataFrame para facilitar la interpretación
        self.cooccurrence_df = pd.DataFrame(self.cooccurrence_matrix, index=self.vocab, columns=self.vocab)

    def generate_text(self, start_word, length=20):
        """
        Genera texto a partir de una palabra inicial usando coocurrencias con selección aleatoria.

        Args:
            start_word (str): Palabra inicial para generar texto.
            length (int): Número de palabras a generar.

        Returns:
            str: Texto generado.
        """
        if start_word not in self.word_to_index:
            raise ValueError("La palabra inicial no está en el vocabulario.")

        current_word = start_word
        generated_text = [current_word]

        for _ in range(length - 1):
            word_idx = self.word_to_index[current_word]
            cooccurrences = self.cooccurrence_matrix[word_idx]

            # Seleccionar las palabras más frecuentes en lugar de solo la más frecuente
            if np.sum(cooccurrences) == 0:
                break  # Si no hay coocurrencias, termina la generación

            # Obtener las palabras con mayor coocurrencia (top 5) y escoger una aleatoriamente
            top_indices = np.argsort(cooccurrences)[-5:]  # Top 5 co-occurrent words
            top_words = [self.vocab[idx] for idx in top_indices]
            current_word = np.random.choice(top_words)
            
            generated_text.append(current_word)

        return ' '.join(generated_text)

def main():
    filepath = '../Gatsby.txt'
    start_marker = "In my younger and more vulnerable years"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK THE GREAT GATSBY ***"

    text_processor = TextProcessor(filepath, start_marker, end_marker)
    text_processor.load_and_clean_text()
    tokens = text_processor.get_tokens()
    print(f"Cantidad de tokens: {len(tokens)}")
    print("\nPrimeros 500 caracteres del texto limpio:")
    print(' '.join(tokens)[:500])

    # Crear el modelo de coocurrencias
    cooccurrence_model = CooccurrenceModel(window_size=4)
    cooccurrence_model.build_cooccurrence_matrix(tokens)

    # Generar texto a partir de una palabra inicial
    start_word = "gatsby"  # Cambia esta palabra para experimentar
    generated_text = cooccurrence_model.generate_text(start_word, length=20)
    print("\nTexto generado con el modelo de coocurrencias:")
    print(generated_text)

    # Opcional: Mostrar la matriz de coocurrencia en forma de tabla (descomenta para ver)
    print("\nMatriz de coocurrencia:")
    # print(cooccurrence_model.cooccurrence_df)

if __name__ == "__main__":
    main()