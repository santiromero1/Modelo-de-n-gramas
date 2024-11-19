
import matplotlib.pyplot as plt
import numpy as np
from TextProcessor import TextProcessor
from models.Ngrams import NgramModel, NgramModelSmoothing, NgramModelAdvanced
from models.Word2Vec import Word2VecModel
from models.FlanT5 import FlanT5TextGenerator
from models.AttentionGPT2 import TransformerPredictor
from models.CooccurrenceModel import CooccurrenceModel
import os
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
import numpy as np
import bert_score

def jaccard_similarity(text1, text2, n=2):
    tokens1 = set(ngrams(word_tokenize(text1.lower()), n))
    tokens2 = set(ngrams(word_tokenize(text2.lower()), n))
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    return len(intersection) / len(union) if len(union) != 0 else 0

def cosine_similarity_metric(text1, text2):
    vectorizer = CountVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    cos_sim = cosine_similarity(vectors)
    return cos_sim[0,1]

def bertscore_metric(generated_text, reference_text, lang='es'):
    P, R, F1 = bert_score.score([generated_text], [reference_text], lang=lang, verbose=False)
    return F1.mean().item()


def main():
    prompt = "In my younger and more vulnerable years"
    ground_truth = (
        "In my younger and more vulnerable years my father gave me some advice "
        "that I've been turning over in my mind ever since. "
        "“Whenever you feel like criticizing anyone,” he told me, “just "
        "remember that all the people in this world haven't had the advantages "
        "that you've had.”"
    )
    text_path = 'Gatsby.txt'

    # Process and clean text
    start_marker = "In my younger and more vulnerable years"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK THE GREAT GATSBY ***"
    text_processor = TextProcessor(text_path, start_marker, end_marker)
    text_processor.load_and_clean_text()
    tokens = text_processor.get_tokens()
    training_text = text_processor.get_text()
    training_segments = text_processor.get_training_segments(segment_length=150)

    # Dictionary to store results
    results = {}

    # Métricas de Similitud
    def compute_similarity_metrics(generated_text, compare_with='ground_truth'):
        if compare_with == 'ground_truth':
            reference_text = ground_truth
        elif compare_with == 'training_segments':
            # Para evitar una comparación desproporcionada, compara con un segmento aleatorio
            # o con el segmento correspondiente si existe una lógica específica
            # Aquí, seleccionaremos el primer segmento para simplificar
            if training_segments:
                reference_text = training_segments[0]
            else:
                reference_text = ""
        else:
            raise ValueError("compare_with debe ser 'ground_truth' o 'training_segments'")
        
        jaccard = jaccard_similarity(generated_text, reference_text, n=2)
        cosine_sim = cosine_similarity_metric(generated_text, reference_text)
        bert_f1 = bertscore_metric(generated_text, reference_text, lang='es')
        return jaccard, cosine_sim, bert_f1
    # FlanT5 Model
    epochs = [3, 10, 100]
    for epoch in epochs:
        print("\n" + "="*80)
        print(f"FlanT5 Text Generation (#Epochs: {epoch})")
        print("="*80)
        flan_model = FlanT5TextGenerator(model_name='google/flan-t5-base')
        flan_model.load_fine_tuned_model(f'./{epoch}_fine_tuned_flan_t5')
        
        # Generate text
        flan_texts = flan_model.generate_text(
            prompt=prompt,
            max_length=300,            # Adjusted based on desired word count
            min_length=150,            # Ensure a minimum length
            num_return_sequences=1,    # Number of different continuations
            temperature=0.7,           # Balanced creativity
            top_k=50,                  # Top-K sampling
            top_p=0.95,                # Nucleus sampling
            num_beams=5,               # Beam search for coherence
            no_repeat_ngram_size=3,    # Prevent repetition
        )
        # For each generated text, calculate accuracy, perplexity and similarity metrics
        for i, flan_text in enumerate(flan_texts, 1):
            # For perplexity, assuming FlanT5 has a calculate_perplexity method
            flan_perplexity = flan_model.calculate_perplexity(flan_text)
            # Compute similarity metrics
            jaccard, cosine_sim, bert_f1 = compute_similarity_metrics(flan_text)
            
            print(f"Generation {i}:\n{flan_text}\n")
            print(f"Perplexity: {flan_perplexity:.2f}")
            print(f"Jaccard Similarity (bigrams): {jaccard:.4f}")
            print(f"Cosine Similarity: {cosine_sim:.4f}")
            print(f"BERTScore F1: {bert_f1:.4f}\n")
            
            results[f'FlanT5_e{epoch}'] = {
                "Generated Text": flan_text,
                "Perplexity": flan_perplexity,
                "Jaccard Similarity (bigrams)": jaccard,
                "Cosine Similarity": cosine_sim,
                "BERTScore F1": bert_f1
            }

    # Basic N-gram Model (n=3, 5, 7)
    for n in [3, 5]:
        print("\n" + "="*80)
        print(f"Basic N-gram Model (n={n})")
        print("="*80)
        ngram_model = NgramModel(n=n)
        ngram_model.build_model(tokens)
        ngram_text = ngram_model.generate_text(prompt, length=150)
        ngram_perplexity = ngram_model.calculate_perplexity(word_tokenize(ngram_text))
        # Compute similarity metrics
        jaccard, cosine_sim, bert_f1 = compute_similarity_metrics(ngram_text)
        
        print("Generated Text:", ngram_text)

        print(f"Perplexity: {ngram_perplexity:.2f}")
        print(f"Jaccard Similarity (bigrams): {jaccard:.4f}")
        print(f"Cosine Similarity: {cosine_sim:.4f}")
        print(f"BERTScore F1: {bert_f1:.4f}\n")
        
        results[f'Basic N-gram (n={n})'] = {
            "Generated Text": ngram_text,
            "Perplexity": ngram_perplexity,
            "Jaccard Similarity (bigrams)": jaccard,
            "Cosine Similarity": cosine_sim,
            "BERTScore F1": bert_f1
        }

    # N-gram Model with Laplace Smoothing (n=3, 5, 7)
    for n in [3, 5]:
        print("\n" + "="*80)
        print(f"N-gram Model with Laplace Smoothing (n={n})")
        print("="*80)
        ngram_smoothing_model = NgramModelSmoothing(n=n)
        ngram_smoothing_model.build_model(tokens)
        ngram_smoothing_text = ngram_smoothing_model.generate_text(prompt, length=150)
        smoothing_perplexity = ngram_smoothing_model.calculate_perplexity(word_tokenize(ngram_smoothing_text))
        # Compute similarity metrics
        jaccard, cosine_sim, bert_f1 = compute_similarity_metrics(ngram_smoothing_text)
        
        print("Generated Text:", ngram_smoothing_text)

        print(f"Perplexity: {smoothing_perplexity:.2f}")
        print(f"Jaccard Similarity (bigrams): {jaccard:.4f}")
        print(f"Cosine Similarity: {cosine_sim:.4f}")
        print(f"BERTScore F1: {bert_f1:.4f}\n")
        
        results[f'N-gram with Laplace Smoothing (n={n})'] = {
            "Generated Text": ngram_smoothing_text,
            "Perplexity": smoothing_perplexity,
            "Jaccard Similarity (bigrams)": jaccard,
            "Cosine Similarity": cosine_sim,
            "BERTScore F1": bert_f1
        }

    # Advanced N-gram Model with different k values
    for k in [5,7]:
        print("\n" + "="*80)
        print(f"Advanced N-gram Model (k={k})")
        print("="*80)
        ngram_advanced_model = NgramModelAdvanced(n=3, k=k)
        ngram_advanced_model.build_model(tokens)
        ngram_advanced_text = ngram_advanced_model.generate_text(prompt, length=150)
        advanced_perplexity = ngram_advanced_model.calculate_perplexity(word_tokenize(ngram_advanced_text))
        # Compute similarity metrics
        jaccard, cosine_sim, bert_f1 = compute_similarity_metrics(ngram_advanced_text)
        
        print("Generated Text:", ngram_advanced_text)
        print(f"Perplexity: {advanced_perplexity:.2f}")
        print(f"Jaccard Similarity (bigrams): {jaccard:.4f}")
        print(f"Cosine Similarity: {cosine_sim:.4f}")
        print(f"BERTScore F1: {bert_f1:.4f}\n")
        
        results[f'Advanced N-gram (k={k})'] = {
            "Generated Text": ngram_advanced_text,
            "Perplexity": advanced_perplexity,
            "Jaccard Similarity (bigrams)": jaccard,
            "Cosine Similarity": cosine_sim,
            "BERTScore F1": bert_f1
        }

    # Word2Vec Model (window=5 and window=10)
    # for window in [5, 10]:
    #     print("\n" + "="*80)
    #     print(f"Word2Vec Model (window={window})")
    #     print("="*80)
    #     word2vec_model = Word2VecModel(load_existing=False)
    #     word2vec_model.train_word2vec([tokens], vector_size=50, window=window)
    #     word2vec_text = word2vec_model.generate_text(prompt, length=150)  # Asumiendo que hay un método para generar texto
    #     word2vec_perplexity = word2vec_model.calculate_perplexity(word_tokenize(word2vec_text))
    #     # Compute similarity metrics
    #     jaccard, cosine_sim, bert_f1 = compute_similarity_metrics(word2vec_text)
        
    #     print("Generated Text:", word2vec_text)
    #     print(f"Perplexity: {word2vec_perplexity:.2f}")
    #     print(f"Jaccard Similarity (bigrams): {jaccard:.4f}")
    #     print(f"Cosine Similarity: {cosine_sim:.4f}")
    #     print(f"BERTScore F1: {bert_f1:.4f}\n")
        
    #     results[f'Word2Vec (window={window})'] = {
    #         "Generated Text": word2vec_text,
    #         "Perplexity": word2vec_perplexity,
    #         "Jaccard Similarity (bigrams)": jaccard,
    #         "Cosine Similarity": cosine_sim,
    #         "BERTScore F1": bert_f1
        # }

    # GPT-2 Transformer Model
    epochs = [3, 10, 100]
    for epoch in epochs:
        print("\n" + "="*80)
        print(f"GPT-2 Transformer Model (#Epochs:{epoch})")
        print("="*80)
        
        gpt2_predictor = TransformerPredictor(model_name='gpt2')
        gpt2_predictor.load_fine_tuned_model(f'./{epoch}_fine_tuned_gpt2')  # Adjust the path if necessary
        
        # Generate text
        gpt2_texts = gpt2_predictor.predict_next_word(prompt, num_predictions=1, max_length=150)
        
        for i, gpt2_text in enumerate(gpt2_texts, 1):
            # Pass the generated text as a string, not tokenized list
            gpt2_perplexity = gpt2_predictor.calculate_perplexity(gpt2_text)
            
            # Compute similarity metrics
            jaccard, cosine_sim, bert_f1 = compute_similarity_metrics(gpt2_text)
            
            print(f"Generation {i}:\n{gpt2_text}\n")
            print(f"Perplexity: {gpt2_perplexity:.2f}")
            print(f"Jaccard Similarity (bigrams): {jaccard:.4f}")
            print(f"Cosine Similarity: {cosine_sim:.4f}")
            print(f"BERTScore F1: {bert_f1:.4f}\n")
            
            results[f'GPT-2_e{epoch}'] = {
                "Generated Text": gpt2_text,
                "Perplexity": gpt2_perplexity,
                "Jaccard Similarity (bigrams)": jaccard,
                "Cosine Similarity": cosine_sim,
                "BERTScore F1": bert_f1
            }

    # Save results to JSON
    with open("generation_results.json", "w", encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print("\nResults saved to generation_results.json")

if __name__ == "__main__":
    main()
