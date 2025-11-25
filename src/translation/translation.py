from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import random
import os
from collections import Counter
from typing import List, Optional, Union, Dict, Any
from abc import ABC, abstractmethod
from tqdm import tqdm


# Base Abstract Class
class BaseTranslator(ABC):
    """Abstract base class for all translation models."""
    
    def __init__(self, device: Optional[str] = None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.is_loaded = False
    
    @abstractmethod
    def load_model(self, model_path: str):
        """Load the translation model."""
        pass
    
    @abstractmethod
    def translate_text(self, text: str, **kwargs) -> str:
        """Translate a single text."""
        pass
    
    @abstractmethod
    def translate_batch(self, texts: List[str], **kwargs) -> List[str]:
        """Translate a batch of texts."""
        pass
    
    def get_model_info(self) -> dict:
        """Get basic model information."""
        return {
            "model_type": self.__class__.__name__,
            "device": self.device,
            "is_loaded": self.is_loaded
        }


# LSTM Model Components
class Vocabulary:
    def __init__(self):
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.freq_threshold = 1

    def __len__(self):
        return len(self.stoi)

    @staticmethod
    def tokenizer(text):
        return text.lower().split()

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenizer(sentence):
                frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer(text)
        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout, pretrained_weights=None):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_dim, emb_dim)

        if pretrained_weights is not None:
            self.embedding.weight.data.copy_(pretrained_weights)
            self.embedding.weight.requires_grad = True

        self.lstm = nn.LSTM(emb_dim, hid_dim, n_layers,
                            dropout=dropout, batch_first=True,
                            bidirectional=True)

        self.dropout = nn.Dropout(dropout)
        self.fc_hidden = nn.Linear(hid_dim * 2, hid_dim)
        self.fc_cell = nn.Linear(hid_dim * 2, hid_dim)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.lstm(embedded)

        hidden_cat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        cell_cat = torch.cat((cell[-2,:,:], cell[-1,:,:]), dim=1)

        hidden = torch.tanh(self.fc_hidden(hidden_cat))
        cell = torch.tanh(self.fc_cell(cell_cat))

        return hidden.unsqueeze(0), cell.unsqueeze(0)


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout, pretrained_weights=None):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)

        if pretrained_weights is not None:
            self.embedding.weight.data.copy_(pretrained_weights)
            self.embedding.weight.requires_grad = False

        self.lstm = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(1)
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        hidden, cell = self.encoder(src)
        input = trg[:, 0]

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[:, t, :] = output

            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[:, t] if teacher_force else top1

        return outputs


# Individual Translator Classes
class LSTMTranslator(BaseTranslator):
    """Custom LSTM-based translator."""
    
    def __init__(self, device: Optional[str] = None, config: Optional[Dict] = None):
        super().__init__(device)
        self.config = config or self._get_default_config()
        self.src_vocab = None
        self.trg_vocab = None
    
    def _get_default_config(self):
        return {
            'ENC_EMB_DIM': 300,
            'DEC_EMB_DIM': 300,
            'HID_DIM': 256,
            'N_LAYERS': 1,
            'ENC_DROPOUT': 0.5,
            'DEC_DROPOUT': 0.5,
            'vocab_data_url': "https://drive.google.com/uc?id=12l_YUBjcAT3CnEYCqmMQwFwTj9ft5N7M"
        }
    
    def load_model(self, model_path: str):
        """Load LSTM model and build vocabularies."""
        try:
            self.src_vocab = Vocabulary()
            self.trg_vocab = Vocabulary()
            
            # Load training data for vocabulary building
            train_df = pd.read_csv(self.config['vocab_data_url'])
            self.src_vocab.build_vocabulary(train_df['text'].tolist())
            self.trg_vocab.build_vocabulary(train_df['english_translation'].tolist())
            
            # Initialize model architecture
            encoder = Encoder(
                len(self.src_vocab), 
                self.config['ENC_EMB_DIM'], 
                self.config['HID_DIM'], 
                self.config['N_LAYERS'], 
                self.config['ENC_DROPOUT']
            )
            decoder = Decoder(
                len(self.trg_vocab), 
                self.config['DEC_EMB_DIM'], 
                self.config['HID_DIM'], 
                self.config['N_LAYERS'], 
                self.config['DEC_DROPOUT']
            )
            self.model = Seq2Seq(encoder, decoder, self.device).to(self.device)
            
            # Load pre-trained weights
            if os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.model.eval()
            
            self.is_loaded = True
            
        except Exception as e:
            raise e
    
    def translate_text(self, text: str, max_length: int = 50, **kwargs) -> str:
        """Translate text using LSTM model."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Please call load_model() first.")
        
        self.model.eval()
        
        if isinstance(text, str):
            tokens = self.src_vocab.numericalize(text)
        else:
            return ""

        tokens = [self.src_vocab.stoi["<SOS>"]] + tokens + [self.src_vocab.stoi["<EOS>"]]
        src_tensor = torch.LongTensor(tokens).unsqueeze(0).to(self.device)

        with torch.no_grad():
            hidden, cell = self.model.encoder(src_tensor)

        trg_indexes = [self.trg_vocab.stoi["<SOS>"]]

        for _ in range(max_length):
            trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(self.device)
            with torch.no_grad():
                output, hidden, cell = self.model.decoder(trg_tensor, hidden, cell)

            pred_token = output.argmax(1).item()
            trg_indexes.append(pred_token)

            if pred_token == self.trg_vocab.stoi["<EOS>"]:
                break

        trg_tokens = [self.trg_vocab.itos[i] for i in trg_indexes]
        return " ".join(trg_tokens[1:-1])
    
    def translate_batch(self, texts: List[str], **kwargs) -> List[str]:
        """Translate batch of texts using LSTM model."""
        return [self.translate_text(text, **kwargs) for text in texts]


class HelsinkiTranslator(BaseTranslator):
    """Helsinki NLP translator."""
    
    def __init__(self, device: Optional[str] = None):
        super().__init__(device)
        self.tokenizer = None
    
    def load_model(self, model_path: str):
        """Load Helsinki NLP model from local path or HuggingFace Hub."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            self.model.to(self.device)
            
            self.is_loaded = True
            
        except Exception as e:
            raise e
    
    def translate_text(self, text: str, max_length: int = 128, **kwargs) -> str:
        """Translate text using Helsinki model."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Please call load_model() first.")
        
        try:
            input_ids = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=128,
                truncation=True
            ).input_ids.to(self.device)
            
            outputs = self.model.generate(
                input_ids,
                max_length=max_length,
                do_sample=False
            )
            
            translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return translation
        except Exception as e:
            return ""
    
    def translate_batch(self, texts: List[str], max_length: int = 128, batch_size: int = 8, **kwargs) -> List[str]:
        """Translate batch of texts using Helsinki model."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Please call load_model() first.")
        
        translations = []
        
        try:
            for text in texts:
                translation = self.translate_text(text, max_length=max_length)
                translations.append(translation)
        except Exception as e:
            pass
            
        return translations


class NLLBTranslator(BaseTranslator):
    """NLLB (No Language Left Behind) translator."""
    
    def __init__(self, device: Optional[str] = None):
        super().__init__(device)
        self.tokenizer = None
    
    def load_model(self, model_path: str):
        """Load NLLB model from local path or HuggingFace Hub."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            self.model.to(self.device)
            
            self.is_loaded = True
            
        except Exception as e:
            raise e
    
    def translate_text(self, text: str, max_length: int = 128, src_lang: str = "ind_Latn", tgt_lang: str = "eng_Latn", **kwargs) -> str:
        """Translate text using NLLB model. Defaults to Indonesian->English."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Please call load_model() first.")
        
        try:
            input_ids = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=128,
                truncation=True
            ).input_ids.to(self.device)
            
            outputs = self.model.generate(
                input_ids,
                forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(tgt_lang),
                max_length=max_length,
                do_sample=False
            )
            
            translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return translation
        except Exception as e:
            return ""
    
    def translate_batch(self, texts: List[str], max_length: int = 128, batch_size: int = 8, **kwargs) -> List[str]:
        """Translate batch of texts using NLLB model."""
        return [self.translate_text(text, max_length=max_length, **kwargs) for text in texts]


class MBartTranslator(BaseTranslator):
    """mBART (Multilingual BART) translator."""
    
    def __init__(self, device: Optional[str] = None):
        super().__init__(device)
        self.tokenizer = None
    
    def load_model(self, model_path: str):
        """Load mBART model from local path or HuggingFace Hub."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            self.model.to(self.device)
            
            self.is_loaded = True
            
        except Exception as e:
            raise e
    
    def translate_text(self, text: str, max_length: int = 128, src_lang: str = "id_ID", tgt_lang: str = "en_XX", **kwargs) -> str:
        """Translate text using mBART model. Defaults to Indonesian->English."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Please call load_model() first.")
        
        try:
            # Handle different language code formats
            if hasattr(self.tokenizer, 'src_lang'):
                self.tokenizer.src_lang = src_lang
            
            input_ids = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=128,
                truncation=True
            ).input_ids.to(self.device)
            
            # Try different ways to get language token ID
            forced_bos_token_id = None
            if hasattr(self.tokenizer, 'lang_code_to_id') and tgt_lang in self.tokenizer.lang_code_to_id:
                forced_bos_token_id = self.tokenizer.lang_code_to_id[tgt_lang]
            elif hasattr(self.tokenizer, 'convert_tokens_to_ids'):
                try:
                    forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(tgt_lang)
                except:
                    pass
            
            if forced_bos_token_id:
                outputs = self.model.generate(
                    input_ids,
                    forced_bos_token_id=forced_bos_token_id,
                    max_length=max_length,
                    do_sample=False
                )
            else:
                outputs = self.model.generate(
                    input_ids,
                    max_length=max_length,
                    do_sample=False
                )
            
            translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return translation
        except Exception as e:
            return ""
    
    def translate_batch(self, texts: List[str], max_length: int = 128, batch_size: int = 8, **kwargs) -> List[str]:
        """Translate batch of texts using mBART model."""
        return [self.translate_text(text, max_length=max_length, **kwargs) for text in texts]


# Main Translation Class
class Translation:
    """Main Translation class that manages different translation models."""
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize the Translation class.
        
        Args:
            device (str, optional): Device to run models on ('cpu' or 'cuda').
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.translators = {}
    
    def load_lstm_model(self, model_path: str = "'/content/drive/MyDrive/Model/LSTM2.pt", config: Optional[Dict] = None):
        """Load LSTM translation model."""
        self.translators['lstm'] = LSTMTranslator(self.device, config)
        self.translators['lstm'].load_model(model_path)
    
    def load_helsinki_model(self, model_name: str = "/content/drive/MyDrive/Model/fine_tuned_helsinki_tourism2"):
        """Load Helsinki NLP translation model."""
        self.translators['helsinki'] = HelsinkiTranslator(self.device)
        self.translators['helsinki'].load_model(model_name)
    
    def load_nllb_model(self, model_name: str = "/content/drive/MyDrive/Model/fine_tuned_nllb_tourism_essential2"):
        """Load NLLB translation model."""
        self.translators['nllb'] = NLLBTranslator(self.device)
        self.translators['nllb'].load_model(model_name)
    
    def load_mbart_model(self, model_name: str = "/content/drive/MyDrive/Model/mbart_fine_tuned_essential2"):
        """Load mBART translation model."""
        self.translators['mbart'] = MBartTranslator(self.device)
        self.translators['mbart'].load_model(model_name)
    
    def translate_text(self, text: str, model_type: str, **kwargs) -> str:
        """
        Translate text using specified model.
        
        Args:
            text (str): Text to translate
            model_type (str): Type of model ('lstm', 'helsinki', 'nllb', 'mbart')
            **kwargs: Additional arguments for translation
            
        Returns:
            str: Translated text
        """
        if model_type not in self.translators:
            raise ValueError(f"Model '{model_type}' not loaded. Available models: {list(self.translators.keys())}")
        
        return self.translators[model_type].translate_text(text, **kwargs)
    
    def translate_batch(self, texts: List[str], model_type: str, **kwargs) -> List[str]:
        """
        Translate batch of texts using specified model.
        
        Args:
            texts (List[str]): List of texts to translate
            model_type (str): Type of model ('lstm', 'helsinki', 'nllb', 'mbart')
            **kwargs: Additional arguments for translation
            
        Returns:
            List[str]: List of translated texts
        """
        if model_type not in self.translators:
            raise ValueError(f"Model '{model_type}' not loaded. Available models: {list(self.translators.keys())}")
        
        return self.translators[model_type].translate_batch(texts, **kwargs)
    
    def get_available_models(self) -> List[str]:
        """Get list of loaded models."""
        return list(self.translators.keys())
    
    def get_model_info(self, model_type: str = None) -> Dict:
        """
        Get information about loaded models.
        
        Args:
            model_type (str, optional): Specific model to get info for. If None, returns all models info.
            
        Returns:
            dict: Model information
        """
        if model_type:
            if model_type not in self.translators:
                raise ValueError(f"Model '{model_type}' not loaded.")
            return self.translators[model_type].get_model_info()
        else:
            return {model_type: translator.get_model_info() 
                    for model_type, translator in self.translators.items()}
    
    def get_translator(self, model_type: str):
        """
        Get specific translator instance.
        
        Args:
            model_type (str): Type of model ('lstm', 'helsinki', 'nllb', 'mbart')
            
        Returns:
            BaseTranslator: The translator instance
        """
        if model_type not in self.translators:
            raise ValueError(f"Model '{model_type}' not loaded. Available models: {list(self.translators.keys())}")
        return self.translators[model_type]