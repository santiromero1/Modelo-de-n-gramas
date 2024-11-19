import os
import gensim
from gensim.models import Word2Vec
from TextProcessor import TextProcessor
import pickle
import numpy as np

class Word2VecModel:
    """
    Modelo de predicción de próxima palabra utilizando embeddings de Word2Vec.
    """
    def __init__(self, save_path='word2vec_embeddings.pkl', load_existing=True):
        """
        Inicializa el modelo y carga o entrena los embeddings de Word2Vec.

        Args:
            save_path (str): Ruta para guardar/cargar los embeddings serializados.
            load_existing (bool): Si True, carga un modelo existente desde el archivo pickle.
        """
        self.save_path = save_path
        if load_existing and os.path.exists(save_path):
            self.load_embeddings_from_pickle(save_path)
        else:
            self.embeddings = None
            self.model = None

    def train_word2vec(self, sentences, vector_size=50, window=5, min_count=1, epochs=100):
        """
        Entrena el modelo Word2Vec en el corpus.

        Args:
            sentences (list): Lista de oraciones (listas de tokens) para entrenar.
            vector_size (int): Dimensión de los embeddings.
            window (int): Máximo de palabras de contexto a considerar.
            min_count (int): Frecuencia mínima de palabras para incluirlas en el modelo.
            epochs (int): Número de iteraciones de entrenamiento.
        """
        print("Entrenando el modelo Word2Vec...")
        self.model = Word2Vec(
            sentences,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            sg=1,  # Skip-gram (1) o CBOW (0)
            epochs=epochs
        )
        self.embeddings = {word: self.model.wv[word] for word in self.model.wv.index_to_key}
        print("Entrenamiento completado. Palabras en vocabulario:", len(self.embeddings))
        self.save_embeddings_to_pickle(self.save_path)

    def save_embeddings_to_pickle(self, save_path):
        """
        Guarda los embeddings en un archivo pickle para cargar rápidamente en el futuro.

        Args:
            save_path (str): Ruta para guardar los embeddings serializados.
        """
        with open(save_path, 'wb') as f:
            pickle.dump(self.embeddings, f)
        print(f"Embeddings guardados en {save_path}.")

    def load_embeddings_from_pickle(self, save_path):
        """
        Carga los embeddings desde un archivo pickle.

        Args:
            save_path (str): Ruta del archivo pickle con los embeddings.
        """
        with open(save_path, 'rb') as f:
            self.embeddings = pickle.load(f)
        print(f"Embeddings cargados desde {save_path}, con {len(self.embeddings)} palabras.")

    def predict_next_word(self, context_words, top_k=5):
        """
        Predice la próxima palabra basándose en las palabras de contexto.

        Args:
            context_words (list): Lista de palabras de contexto.
            top_k (int): Número de palabras más probables a retornar.

        Returns:
            list: Lista de palabras predichas ordenadas por similitud.
        """
        if not self.embeddings:
            raise ValueError("El modelo debe ser entrenado o cargado antes de hacer predicciones.")
        
        # Calcular el vector de contexto promedio
        context_vectors = [self.embeddings[word] for word in context_words if word in self.embeddings]
        if not context_vectors:
            return []

        context_vector = sum(context_vectors) / len(context_vectors)
        
        # Calcular las similitudes con todas las palabras del vocabulario
        similarities = {}
        for word in self.embeddings:
            similarities[word] = self.cosine_similarity(context_vector, self.embeddings[word])

        # Obtener las palabras con mayor similitud
        top_words = sorted(similarities, key=similarities.get, reverse=True)[:top_k]
        return top_words
    
    def calculate_perplexity(self, tokens):
        log_prob_sum = 0
        for i in range(len(tokens) - 1):
            context = tokens[:i+1]
            next_word = tokens[i + 1]
            predictions = self.predict_next_word(context, top_k=5)
            prob = 1 / (predictions.index(next_word) + 1) if next_word in predictions else 1e-10
            log_prob_sum += np.log(prob)
        return np.exp(-log_prob_sum / (len(tokens) - 1))

    @staticmethod
    def cosine_similarity(vec1, vec2):
        """
        Calcula la similitud de coseno entre dos vectores.

        Args:
            vec1 (np.array): Primer vector.
            vec2 (np.array): Segundo vector.

        Returns:
            float: Similitud de coseno entre los dos vectores.
        """
        dot_product = np.dot(vec1, vec2)
        norm_a = np.linalg.norm(vec1)
        norm_b = np.linalg.norm(vec2)
        return dot_product / (norm_a * norm_b) if norm_a != 0 and norm_b != 0 else 0.0

def main():
    # Procesar el texto
    filepath = 'Gatsby.txt'
    start_marker = "In my younger and more vulnerable years"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK THE GREAT GATSBY ***"

    text_processor = TextProcessor(filepath, start_marker, end_marker)
    text_processor.load_and_clean_text()
    tokens = text_processor.get_tokens()
    print(f"Cantidad de tokens: {len(tokens)}")

    # Preparar el corpus como lista de oraciones para entrenamiento Word2Vec
    sentences = [tokens]  # Aquí, cada oración es una lista de tokens

    # Inicializar y entrenar el modelo Word2Vec
    word2vec_model = Word2VecModel(save_path='word2vec_embeddings.pkl', load_existing=False)
    word2vec_model.train_word2vec(sentences)

    # Predicción de la próxima palabra
    context = "in my younger and more"
    context_words = context.lower().split()
    predicted_words = word2vec_model.predict_next_word(context_words, top_k=5)
    print("\nPalabras predichas para el contexto '{}':".format(context))
    print(predicted_words)

if __name__ == "__main__":
    main()
