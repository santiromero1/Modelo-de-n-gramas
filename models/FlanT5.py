import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset
import pandas as pd
import numpy as np
import os

# Set environment variable for CUDA memory allocation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

class FlanT5TextGenerator:
    """
    Class for fine-tuning and using FLAN-T5 for text generation.
    """
    def __init__(self, model_name='google/flan-t5-base', max_length=256):
        """
        Initialize the FLAN-T5 model for fine-tuning and text generation.

        Args:
            model_name (str): Hugging Face model name.
            max_length (int): Maximum sequence length.
        """
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.gradient_checkpointing_enable()  # Enable gradient checkpointing
        self.max_length = max_length
        # Move model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def prepare_data(self, text, chunk_size=50):
        """
        Prepare text data for fine-tuning by creating prompt-completion pairs.

        Args:
            text (str): Text corpus as a single string.
            chunk_size (int): Number of words in each prompt-completion pair.

        Returns:
            Dataset: Hugging Face Dataset for training.
        """
        words = text.split()
        examples = []
        for i in range(0, len(words) - chunk_size, chunk_size):
            prompt = " ".join(words[i:i + chunk_size])
            completion = " ".join(words[i + 1: i + chunk_size + 1])
            examples.append({"prompt": prompt, "completion": completion})

        return Dataset.from_pandas(pd.DataFrame(examples))

    def fine_tune(self, dataset, output_dir='./fine_tuned_flan_t5', num_train_epochs=3, batch_size=1, gradient_accumulation_steps=4):
        """
        Fine-tune FLAN-T5 on the dataset.

        Args:
            dataset (Dataset): Hugging Face Dataset with 'prompt' and 'completion' columns.
            output_dir (str): Directory to save the fine-tuned model.
            num_train_epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            gradient_accumulation_steps (int): Number of steps to accumulate gradients.
        """
        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)

        def tokenize_batch(batch):
            inputs = self.tokenizer(batch["prompt"], padding="max_length", truncation=True, max_length=self.max_length)
            targets = self.tokenizer(batch["completion"], padding="max_length", truncation=True, max_length=self.max_length)
            batch["input_ids"] = inputs.input_ids
            batch["attention_mask"] = inputs.attention_mask
            batch["labels"] = targets.input_ids
            return batch

        tokenized_dataset = dataset.map(tokenize_batch, batched=True)

        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            save_steps=10_000,
            save_total_limit=2,
            logging_steps=500,
            prediction_loss_only=True,
            fp16=True,  # Enable mixed precision
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )

        trainer.train()
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def load_fine_tuned_model(self, model_dir):
        """
        Load a previously fine-tuned model.

        Args:
            model_dir (str): Directory where the fine-tuned model is saved.
        """
        self.model = T5ForConditionalGeneration.from_pretrained(model_dir)
        self.tokenizer = T5Tokenizer.from_pretrained(model_dir)
        self.model.to(self.device)

    def generate_text(
        self,
        prompt,
        max_length=250,
        min_length=150,
        num_return_sequences=3,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        num_beams=5,
        no_repeat_ngram_size=3,
    ):
        """
        Generate a coherent and detailed paragraph continuation using FLAN-T5 based on a given prompt.

        Args:
            prompt (str): Initial text prompt.
            max_length (int): Maximum number of tokens in generated text.
            min_length (int): Minimum number of tokens in generated text.
            num_return_sequences (int): Number of generated sequences.
            temperature (float): Sampling temperature. Lower values make the output more deterministic.
            top_k (int): The number of highest probability vocabulary tokens to keep for top-k-filtering.
            top_p (float): Cumulative probability for nucleus sampling.
            num_beams (int): Number of beams for beam search (higher for better quality).
            no_repeat_ngram_size (int): Prevents the model from repeating n-grams of this size.

        Returns:
            List[str]: List of generated text sequences.
        """
        # Define the instruction and combine with the user-provided prompt
        instruction = "Continue the following sentence to form a coherent and detailed paragraph:\n\n"
        full_prompt = instruction + prompt

        # Tokenize the input prompt
        input_ids = self.tokenizer.encode(full_prompt, return_tensors="pt").to(self.device)

        # Generate the output sequences
        outputs = self.model.generate(
            input_ids=input_ids,
            max_length=max_length,
            min_length=min_length,
            num_return_sequences=num_return_sequences,
            num_beams=num_beams,
            no_repeat_ngram_size=no_repeat_ngram_size,
            early_stopping=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,  # Enable sampling for diversity
        )

        # Decode the generated sequences
        generated_texts = [
            self.tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for output in outputs
        ]

        # Extract only the continuation part by removing the prompt
        continuations = [text[len(full_prompt):].strip() for text in generated_texts]

        return continuations

    def calculate_perplexity(self, text):
        """
        Calculate perplexity for Flan-T5 model.
        Args:
            text (str): Input text for which perplexity is calculated.
        Returns:
            float: Perplexity score.
        """
        if isinstance(text, list):  # Convert tokenized text to string
            text = ' '.join(text)

        # Tokenize with padding and truncation
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )

        input_ids = encoding.input_ids.to(self.model.device)
        attention_mask = encoding.attention_mask.to(self.model.device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss.item()

        return np.exp(loss)

def main():
    # Load your book text
    with open("./Gatsby.txt", "r", encoding="utf-8") as file:
        text = file.read()

    # Initialize the text generator with a smaller model and reduced max_length
    generator = FlanT5TextGenerator(model_name='google/flan-t5-base', max_length=256)

    # Load an existing fine-tuned model if available
    model_dir = './fine_tuned_flan_t5'
    if os.path.exists(model_dir):
        print("Loading existing fine-tuned model...")
        generator.load_fine_tuned_model(model_dir)
    else:
        print("Preparing data for fine-tuning...")
        # Prepare data for fine-tuning with smaller chunk size
        dataset = generator.prepare_data(text, chunk_size=50)
        print("Starting fine-tuning...")
        # Fine-tune with reduced batch size and gradient accumulation
        generator.fine_tune(dataset, output_dir=model_dir, num_train_epochs=5, batch_size=1, gradient_accumulation_steps=4)
        print("Fine-tuning completed and model saved.")

    # Use the fine-tuned model to generate text
    prompt = "In my younger and more vulnerable years,"
    generated_texts = generator.generate_text(
        prompt=prompt,
        max_length=200,            # Adjusted based on desired word count
        min_length=150,            # Ensure a minimum length
        num_return_sequences=3,    # Number of different continuations
        temperature=0.7,           # Balanced creativity
        top_k=50,                  # Top-K sampling
        top_p=0.95,                # Nucleus sampling
        num_beams=5,               # Beam search for coherence
        no_repeat_ngram_size=3,    # Prevent repetition
    )

    print("\nGenerated Texts:")
    for i, text in enumerate(generated_texts, 1):
        print(f"Generation {i}: {text}\n")


if __name__ == "__main__":
    main()
