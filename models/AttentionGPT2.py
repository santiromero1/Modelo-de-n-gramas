from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from transformers import TextDataset, DataCollatorForLanguageModeling
import torch
import os
import numpy as np

class TransformerPredictor:
    def __init__(self, model_name='gpt2', max_length=128):
        """
        Initialize the transformer model with GPT-2.

        Args:
            model_name (str): Hugging Face model name to use, such as 'gpt2' or 'gpt2-medium'.
            max_length (int): Maximum length for input text sequences.
        """
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        # Set pad_token to eos_token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # Ensure the model knows about the pad_token_id
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.config.pad_token_id = self.tokenizer.eos_token_id
        
        self.max_length = max_length
    def fine_tune(self, text_file_path, output_dir='./fine_tuned_model', num_train_epochs=3, batch_size=4):
        """
        Fine-tune the model on a text corpus.

        Args:
            text_file_path (str): Path to the text file to use for training.
            output_dir (str): Directory to save the fine-tuned model.
            num_train_epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
        """
        dataset = TextDataset(
            tokenizer=self.tokenizer,
            file_path=text_file_path,
            block_size=self.max_length
        )
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )

        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=batch_size,
            save_steps=10_000,
            save_total_limit=2,
            logging_steps=500,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset,
        )

        trainer.train()
        trainer.save_model(output_dir)
            # Ensure the tokenizer is saved as well
        self.tokenizer.save_pretrained(output_dir)

    def load_fine_tuned_model(self, model_dir):
        """
        Load a previously fine-tuned model.

        Args:
            model_dir (str): Directory where the fine-tuned model is saved.
        """
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
        self.model = GPT2LMHeadModel.from_pretrained(model_dir)
        self.model.eval()

    def predict_next_word(self, prompt, num_predictions=5, max_length=50, temperature=0.7):
        """
        Predict the next word given a text prompt.

        Args:
            prompt (str): Text prompt to generate the next word.
            num_predictions (int): Number of sequences to generate.
            max_length (int): Maximum length of the generated text including the prompt.
            temperature (float): Sampling temperature; higher values make output more random.

        Returns:
            List[str]: List of generated sequences.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = inputs.to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=max_length,
                temperature=temperature,
                num_return_sequences=num_predictions,
                top_k=50,
                top_p=0.95,
                do_sample=True
            )

        predictions = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        return predictions
    def calculate_perplexity(self, text):
        """Calculate perplexity for GPT-2 model."""
        # Ensure text is a string
        if isinstance(text, list):
            text = ' '.join(text)
        
        # Tokenize with padding and truncation
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length  # Ensure max_length is respected
        )
        
        input_ids = encoding.input_ids.to(self.model.device)
        attention_mask = encoding.attention_mask.to(self.model.device)
        
        with torch.no_grad():
            loss = self.model(input_ids, attention_mask=attention_mask, labels=input_ids).loss.item()
        
        return np.exp(loss)


def main():
    # Define file path to the text corpus
    text_file_path = './Gatsby.txt'
    
    # Initialize and fine-tune the Transformer model
    model_name = 'gpt2'  # Can be changed to 'gpt2-medium' for a larger model
    predictor = TransformerPredictor(model_name=model_name)
    
    # Fine-tune the model with the provided text file
    # (adjust num_train_epochs based on the size of your dataset and compute power)
    predictor.fine_tune(text_file_path, output_dir='./fine_tuned_gpt2', num_train_epochs=100, batch_size=5)
    
    # Alternatively, if the model is already fine-tuned, load it
    # predictor.load_fine_tuned_model('./fine_tuned_gpt2')
    
    # Prompt to test the next word prediction
    prompt = "In my younger and more vulnerable years"
    predictions = predictor.predict_next_word(prompt, num_predictions=4, max_length=50)
    
    print("\nPredictions:")
    for i, prediction in enumerate(predictions, 1):
        print(f"Prediction {i}: {prediction}")

if __name__ == "__main__":
    main()
