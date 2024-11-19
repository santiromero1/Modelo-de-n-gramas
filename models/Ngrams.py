import random
from nltk import ngrams
from collections import defaultdict
from TextProcessor import TextProcessor
import math
import numpy as np


class NgramModel:
    """
    Modelo básico de n-gramas sin suavizado.
    """
    def __init__(self, n):
        """
        Inicializa el modelo de n-gramas.

        Args:
            n (int): Tamaño de los n-gramas (e.g., 2 para bigramas).
        """
        self.n = n
        self.model = defaultdict(int)

    def build_model(self, tokens):
        """
        Construye el modelo de n-gramas a partir de una lista de tokens.

        Args:
            tokens (list): Lista de tokens del corpus.
        """
        for ngram in ngrams(tokens, self.n):
            self.model[ngram] += 1

    def generate_text(self, start_sequence, length=50):
        """
        Generates text from a seed sequence using the n-gram model.

        Args:
            start_sequence (str): Initial sequence of words.
            length (int): Number of words to generate.

        Returns:
            str: Generated text.
        """
        start_tokens = start_sequence.lower().split()[-(self.n - 1):]  # Adjusts to take only the last n-1 words if longer
        if len(start_tokens) != self.n - 1:
            raise ValueError(f"La semilla debe tener {self.n - 1} palabras.")

        generated = start_tokens.copy()
        for _ in range(length):
            context = tuple(generated[-(self.n - 1):])
            candidates = {ngram[-1]: freq for ngram, freq in self.model.items() if ngram[:-1] == context}
            if not candidates:
                break
            next_word = max(candidates, key=candidates.get)
            generated.append(next_word)
        return ' '.join(generated)
    def calculate_perplexity(self, tokens):
        n = self.n
        log_prob_sum = 0
        num_words = len(tokens) - n + 1
        for i in range(num_words):
            context = tuple(tokens[i:i + n - 1])
            word = tokens[i + n - 1]
            prob = self.model.get(context + (word,), 1e-10)  # Handle unseen words
            log_prob_sum += np.log(prob)
        return np.exp(-log_prob_sum / num_words)

class NgramModelSmoothing:
    """
    Modelo de n-gramas con suavizado de Laplace.
    """
    def __init__(self, n):
        """
        Inicializa el modelo de n-gramas con suavizado de Laplace.

        Args:
            n (int): Tamaño de los n-gramas.
        """
        self.n = n
        self.model = defaultdict(int)
        self.vocabulary = set()
    
    def build_model(self, tokens):
        """
        Construye el modelo de n-gramas y aplica suavizado de Laplace.

        Args:
            tokens (list): Lista de tokens del corpus.
        """
        self.vocabulary = set(tokens)
        for ngram in ngrams(tokens, self.n):
            self.model[ngram] += 1
        
        # Aplicar suavizado de Laplace
        self.apply_laplace_smoothing()

    def apply_laplace_smoothing(self):
        """
        Aplica suavizado de Laplace al modelo de n-gramas.
        """
        vocab_size = len(self.vocabulary)
        for ngram in self.model:
            self.model[ngram] += 1  # Sumar 1 a cada n-grama observado
        
        # Ajustar total de posibles n-gramas para la normalización
        total_possible_ngrams = (vocab_size) ** self.n
        for ngram in self.model:
            self.model[ngram] /= (total_possible_ngrams)

    def generate_text(self, start_sequence, length=50):
        """
        Genera texto a partir de una semilla utilizando el modelo de n-gramas con suavizado de Laplace.

        Args:
            start_sequence (str): Secuencia inicial de palabras (n-1 palabras).
            length (int): Número de palabras a generar.

        Returns:
            str: Texto generado.
        """
        start_tokens = start_sequence.lower().split()[-(self.n - 1):]
        if len(start_tokens) != self.n - 1:
            raise ValueError(f"La semilla debe tener {self.n - 1} palabras.")

        generated = start_tokens.copy()
        for _ in range(length):
            context = tuple(generated[-(self.n - 1):])
            candidates = {ngram[-1]: freq for ngram, freq in self.model.items() if ngram[:-1] == context}
            if not candidates:
                break
            
            next_word = random.choices(list(candidates.keys()), weights=list(candidates.values()))[0]
            generated.append(next_word)
        
        return ' '.join(generated)
    def calculate_perplexity(self, tokens):
        n = self.n
        log_prob_sum = 0
        num_words = len(tokens) - n + 1
        for i in range(num_words):
            context = tuple(tokens[i:i + n - 1])
            word = tokens[i + n - 1]
            prob = self.model.get(context + (word,), 1e-10)  # Handle unseen words
            log_prob_sum += np.log(prob)
        return np.exp(-log_prob_sum / num_words)

class NgramModelAdvanced:
    """
    Modelo avanzado de n-gramas con selección aleatoria entre las k palabras más probables.
    """
    def __init__(self, n, k=5):
        """
        Inicializa el modelo avanzado de n-gramas.

        Args:
            n (int): Tamaño de los n-gramas.
            k (int): Número de palabras más probables entre las cuales seleccionar aleatoriamente.
        """
        self.n = n
        self.k = k  # Número de palabras para la selección aleatoria
        self.model = defaultdict(lambda: defaultdict(int))
        self.vocabulary = set()

    def build_model(self, tokens):
        """
        Construye el modelo de n-gramas y el vocabulario a partir de una lista de tokens.

        Args:
            tokens (list): Lista de tokens del corpus.
        """
        self.vocabulary = set(tokens)
        for i in range(len(tokens) - self.n + 1):
            contexto = tuple(tokens[i:i + self.n -1])
            palabra_siguiente = tokens[i + self.n -1]
            self.model[contexto][palabra_siguiente] += 1

    def smooth_model(self, delta=1):
        """
        Aplica suavizado Laplace (add-one) al modelo.

        Args:
            delta (int, opcional): Valor a añadir a cada conteo. Por defecto es 1.
        """
        for contexto in self.model:
            for palabra in self.vocabulary:
                self.model[contexto][palabra] += delta

    def generate_text(self, start_sequence, length=50):
        """
        Genera texto a partir de una semilla utilizando el modelo avanzado de n-gramas.

        Args:
            start_sequence (str): Secuencia inicial de palabras.
            length (int): Número de palabras a generar.

        Returns:
            str: Texto generado.
        """
        # Use only the last n-1 words of the start_sequence
        start_tokens = start_sequence.lower().split()[-(self.n - 1):]
        if len(start_tokens) != self.n - 1:
            raise ValueError(f"La semilla debe tener {self.n - 1} palabras.")

        generated = start_tokens.copy()
        for _ in range(length):
            context = tuple(generated[-(self.n - 1):])
            palabras_posibles = self.model.get(context, None)
            if not palabras_posibles:
                break
            # Obtener las k palabras más probables
            palabras_ordenadas = sorted(palabras_posibles.items(), key=lambda item: item[1], reverse=True)
            top_k = palabras_ordenadas[:self.k]
            if not top_k:
                break
            # Seleccionar aleatoriamente una de las top_k palabras
            siguiente_palabra = random.choice(top_k)[0]
            generated.append(siguiente_palabra)
        return ' '.join(generated)
    def calculate_perplexity(self, tokens):
        n = self.n
        log_prob_sum = 0
        num_words = len(tokens) - n + 1
        for i in range(num_words):
            context = tuple(tokens[i:i + n - 1])
            word = tokens[i + n - 1]
            prob = self.model.get(context + (word,), 1e-10)  # Handle unseen words
            log_prob_sum += np.log(prob)
        return np.exp(-log_prob_sum / num_words)

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

    # 2. Modelo Básico de n-gramas
    print("\n" + "="*80)
    print("Modelo Básico de n-gramas")
    print("="*80)
    n = 3  # Trigramas
    start_sequence = "in my younger and more"

    basic_ngram_model = NgramModel(n)
    basic_ngram_model.build_model(tokens)
    generated_text_basic = basic_ngram_model.generate_text(start_sequence, length=100)
    print(start_sequence + "\n" +generated_text_basic)

    # 3. Modelo de n-gramas con Suavizado de Laplace
    print("\n" + "="*80)
    print("Modelo de n-gramas con Suavizado de Laplace")
    print("="*80)
    n = 3  # Trigramas
    start_sequence = "in my younger and more"

    smoothing_ngram_model = NgramModelSmoothing(n)
    smoothing_ngram_model.build_model(tokens)
    generated_text = smoothing_ngram_model.generate_text(start_sequence, length=50)

    print(start_sequence + "\n" +generated_text)

    # 4. Modelo Avanzado de n-gramas con Selección Aleatoria entre las k Palabras Más Probables
    print("\n" + "="*80)
    print("Modelo Avanzado de n-gramas con Selección Aleatoria")
    print("="*80)
    n_advanced = 3  # Trigramas
    k = 5  # Número de palabras para la selección aleatoria
    start_sequence_advanced = "in my younger and more"

    advanced_ngram_model = NgramModelAdvanced(n_advanced, k)
    advanced_ngram_model.build_model(tokens)
    advanced_ngram_model.smooth_model(delta=1)  # Aplicar suavizado si es necesario
    generated_text_advanced = advanced_ngram_model.generate_text(start_sequence_advanced, length=100)
    print(start_sequence + "\n" +generated_text_advanced)
    
    print("="*80)
    
    test_tokens = ["in", "my", "younger", "and", "more"]  # tokens de prueba
    perplexity_base = calculate_perplexity(basic_ngram_model, test_tokens)
    print("Perplejidad del modelo Ngram Base:", perplexity_base)
    perplexity_smooth = calculate_perplexity(smoothing_ngram_model, test_tokens)
    print("Perplejidad del modelo Ngram Smooth:", perplexity_smooth)
    perplexity_avanzado = calculate_perplexity(advanced_ngram_model, test_tokens)
    print("Perplejidad del modelo Ngram Avanzado:", perplexity_avanzado)


if __name__ == "__main__":
    main()