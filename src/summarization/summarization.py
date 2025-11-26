import os
import torch
from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    PegasusTokenizer,
    PegasusForConditionalGeneration,
)
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer as SumyTokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sentence_splitter import SentenceSplitter
import nltk
import pandas as pd

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)


class Summarization:
    def __init__(self):
        """Initialize the Summarization class with model configurations."""
        self.MAX_INPUT_LENGTH = 512
        self.MAX_TARGET_LENGTH = 128
        self.loaded_models = {}
        
        # Get the base directory (src/summarization -> project root)
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.models_dir = os.path.join(base_dir, "models")
        
        # Model paths
        self.MODEL_PATHS = {
            "BART_FULL": os.path.join(self.models_dir, "BART_FULL"),
            "BART_SAMPLED": os.path.join(self.models_dir, "BART_SAMPLED"),
            "PEGASUS_FULL": os.path.join(self.models_dir, "PEGASUS_FULL"),
            "PEGASUS_SAMPLED": os.path.join(self.models_dir, "PEGASUS_SAMPLED"),
        }
        
        # Base model checkpoints for zero-shot
        self.BASE_MODELS = {
            "BART": "facebook/bart-base",
            "PEGASUS": "google/pegasus-large",
        }

    def extract_first_sentence(self, text):
        """Extract the first sentence from text as a simple baseline summary."""
        if pd.isna(text) or not text.strip():
            return ""
        sentences = nltk.sent_tokenize(text)
        return sentences[0].strip() if sentences else ""

    def summarize_textrank(self, text, sentence_count=2):
        """
        Generate extractive summary using TextRank algorithm.
        
        Args:
            text (str): Input text to summarize
            sentence_count (int): Number of sentences in summary
            
        Returns:
            str: Extractive summary
        """
        if pd.isna(text) or not text.strip():
            return self.extract_first_sentence(text)
        
        try:
            parser = PlaintextParser.from_string(text, SumyTokenizer("english"))
            summarizer = TextRankSummarizer()
            splitter = SentenceSplitter(language='en')
            sentences = splitter.split(text)
            num_sentences = min(len(sentences), sentence_count)
            summary = summarizer(parser.document, num_sentences)
            return " ".join([str(s) for s in summary])
        except Exception:
            return self.extract_first_sentence(text)

    def load_model(self, model_type, model_name):
        """
        Load a summarization model (BART or PEGASUS, fine-tuned or zero-shot).
        
        Args:
            model_type (str): 'BART' or 'PEGASUS'
            model_name (str): Specific model variant (e.g., 'BART_FULL', 'BART_SAMPLED')
            
        Returns:
            tuple: (model, tokenizer)
        """
        cache_key = model_name if model_name else model_type
        
        if cache_key in self.loaded_models:
            return self.loaded_models[cache_key]
        
        # Determine model path
        if model_name and model_name in self.MODEL_PATHS:
            model_path = self.MODEL_PATHS[model_name]
            if not os.path.exists(model_path):
                print(f"Warning: Fine-tuned model not found at {model_path}. Using base model instead.")
                model_path = self.BASE_MODELS[model_type]
        else:
            model_path = self.BASE_MODELS[model_type]
        
        # Load tokenizer and model
        if model_type == "BART":
            tokenizer = BartTokenizer.from_pretrained(self.BASE_MODELS["BART"])
            model = BartForConditionalGeneration.from_pretrained(model_path)
        elif model_type == "PEGASUS":
            tokenizer = PegasusTokenizer.from_pretrained(self.BASE_MODELS["PEGASUS"])
            model = PegasusForConditionalGeneration.from_pretrained(model_path)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Move to GPU if available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        model.eval()
        
        self.loaded_models[cache_key] = (model, tokenizer)
        return model, tokenizer

    def generate_summary_transformer(self, text, model_type, model_name=None):
        """
        Generate abstractive summary using transformer models.
        
        Args:
            text (str): Input text to summarize
            model_type (str): 'BART' or 'PEGASUS'
            model_name (str): Specific model variant (optional)
            
        Returns:
            str: Generated summary
        """
        if pd.isna(text) or not text.strip():
            return ""
        
        model, tokenizer = self.load_model(model_type, model_name)
        
        # Tokenize input
        inputs = tokenizer(
            text,
            max_length=self.MAX_INPUT_LENGTH,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        
        # Move to device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate summary
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=self.MAX_TARGET_LENGTH,
                min_length=10,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True,
            )
        
        # Decode and return
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary

    def summarize(self, text, method='best'):
        """
        Generate summary using specified method.
        
        Args:
            text (str): Input text to summarize
            method (str): Summarization method
                - 'first_sentence': Extract first sentence
                - 'textrank': TextRank extractive summarization
                - 'bart_zero_shot': BART base model (zero-shot)
                - 'bart_full': BART fine-tuned on full dataset
                - 'bart_sampled': BART fine-tuned on sampled dataset
                - 'pegasus_zero_shot': PEGASUS base model (zero-shot)
                - 'pegasus_full': PEGASUS fine-tuned on full dataset
                - 'pegasus_sampled': PEGASUS fine-tuned on sampled dataset
                - 'best': Use best performing model (BART_SAMPLED)
                
        Returns:
            str: Generated summary
        """
        if pd.isna(text) or not text.strip():
            return ""
        
        method = method.lower()
        
        # Extractive methods
        if method == 'first_sentence':
            return self.extract_first_sentence(text)
        elif method == 'textrank':
            return self.summarize_textrank(text)
        
        # BART models
        elif method == 'bart_zero_shot':
            return self.generate_summary_transformer(text, 'BART', None)
        elif method == 'bart_full':
            return self.generate_summary_transformer(text, 'BART', 'BART_FULL')
        elif method == 'bart_sampled':
            return self.generate_summary_transformer(text, 'BART', 'BART_SAMPLED')
        
        # PEGASUS models
        elif method == 'pegasus_zero_shot':
            return self.generate_summary_transformer(text, 'PEGASUS', None)
        elif method == 'pegasus_full':
            return self.generate_summary_transformer(text, 'PEGASUS', 'PEGASUS_FULL')
        elif method == 'pegasus_sampled':
            return self.generate_summary_transformer(text, 'PEGASUS', 'PEGASUS_SAMPLED')
        
        # Best model (based on experiment results)
        elif method == 'best':
            return self.generate_summary_transformer(text, 'BART', 'BART_SAMPLED')
        
        else:
            raise ValueError(f"Unknown summarization method: {method}. "
                           f"Available methods: first_sentence, textrank, "
                           f"bart_zero_shot, bart_full, bart_sampled, "
                           f"pegasus_zero_shot, pegasus_full, pegasus_sampled, best")

    def get_available_models(self):
        """
        Get list of available models with their status.
        
        Returns:
            dict: Model availability status
        """
        available = {
            'extractive': ['first_sentence', 'textrank'],
            'bart': {
                'zero_shot': True,
                'full': os.path.exists(self.MODEL_PATHS.get('BART_FULL', '')),
                'sampled': os.path.exists(self.MODEL_PATHS.get('BART_SAMPLED', ''))
            },
            'pegasus': {
                'zero_shot': True,
                'full': os.path.exists(self.MODEL_PATHS.get('PEGASUS_FULL', '')),
                'sampled': os.path.exists(self.MODEL_PATHS.get('PEGASUS_SAMPLED', ''))
            }
        }
        return available
