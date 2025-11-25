import time
import sys
import os
import random
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from translation.translation import Translation
from summarization.summarization import Summarization

VALID_ASPECTS = [
    'place', 'spot', 'view', 'price', 'quietness', 'food', 'drink', 'maintenance',
    'staff', 'peacefulness', 'road', 'air', 'water', 'comfort', 'photo', 'music',
    'restroom', 'parking', 'taxi', 'spaciousness', 'light', 'accessibility',
    'cleanliness', 'atmosphere', 'management', 'service', 'facility',
    'location', 'weather', 'crowd', 'safety', 'entrance', 'ticket', 'guide',
    'information', 'wifi', 'internet', 'scenery', 'beauty', 'nature',
    'architecture', 'history', 'culture', 'tradition', 'souvenir', 'shopping',
    'accommodation', 'hotel', 'restaurant', 'cafe', 'transportation',
    'walking', 'hiking', 'trail', 'path', 'signage', 'direction', 'map',
    'security', 'noise', 'crowdedness', 'queue', 'waiting', 'opening_hours',
    'tour', 'activity', 'entertainment', 'attraction', 'monument', 'museum',
    'garden', 'beach', 'mountain', 'lake', 'river', 'bridge', 'temple',
    'religious', 'spiritual', 'calm', 'serenity', 'beauty', 'landscape'
]


class ABSA:
    def __init__(self):
        self.valid_aspects = VALID_ASPECTS
        self.aspect_label_map = {asp: idx for idx, asp in enumerate(self.valid_aspects)}
        self.reverse_aspect_label_map = {idx: asp for idx, asp in enumerate(self.valid_aspects)}
        self.sentiment_label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        self.reverse_sentiment_label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
        self.real_aspect_label_map = {'O': 0, 'B-ASPECT': 1, 'I-ASPECT': 2}
        self.reverse_real_aspect_label_map = {0: 'O', 1: 'B-ASPECT', 2: 'I-ASPECT'}
        self.loaded_models = {}

    def load_transformer_model(self, model_name, task):
        cache_key = f"{model_name}_{task}"
        if cache_key in self.loaded_models:
            return self.loaded_models[cache_key]
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, "models", "absa", f"{model_name}_{task}")

        if task == 'task1':
            # Token classification model for aspect extraction
            model = AutoModelForTokenClassification.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            # Sequence classification model for aspect/sentiment classification
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.loaded_models[cache_key] = (model, tokenizer)
        return model, tokenizer

    def extract_real_aspects_rule_based(self, text):
        """Rule-based extraction of real aspects from text"""
        blob = TextBlob(text)
        real_aspects = []

        # Extract noun phrases
        for np in blob.noun_phrases:
            if len(np.split()) <= 3:
                real_aspects.append(np.strip())

        # Extract individual nouns
        for word, pos in blob.tags:
            if pos.startswith('NN') and len(word) > 2:
                real_aspects.append(word.lower())

        # Remove duplicates
        seen = set()
        unique_aspects = []
        for asp in real_aspects:
            if asp not in seen:
                seen.add(asp)
                unique_aspects.append(asp)

        return unique_aspects if unique_aspects else []

    def classify_aspect_rule_based(self, sentence, real_aspect):
        real_asp_lower = real_aspect.lower()

        # Direct match
        if real_asp_lower in [va.lower() for va in self.valid_aspects]:
            idx = [va.lower() for va in self.valid_aspects].index(real_asp_lower)
            return self.valid_aspects[idx]

        # Partial match
        for valid_asp in self.valid_aspects:
            if valid_asp.lower() in real_asp_lower or real_asp_lower in valid_asp.lower():
                return valid_asp

        return 'place'  # Default

    def load_tfidf_lr_models(self):
        """Load TF-IDF + LR models"""
        if 'tfidf_lr' in self.loaded_models:
            return self.loaded_models['tfidf_lr']

        try:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            models_dir = os.path.join(base_dir, "models", "absa")

            task2_model = joblib.load(os.path.join(models_dir, "tfidf_lr_task2_model.pkl"))
            task2_vectorizer = joblib.load(os.path.join(models_dir, "tfidf_lr_task2_vectorizer.pkl"))
            task3_model = joblib.load(os.path.join(models_dir, "tfidf_lr_task3_model.pkl"))
            task3_vectorizer = joblib.load(os.path.join(models_dir, "tfidf_lr_task3_vectorizer.pkl"))
            self.loaded_models['tfidf_lr'] = (task2_model, task2_vectorizer, task3_model, task3_vectorizer)
            return task2_model, task2_vectorizer, task3_model, task3_vectorizer
        except Exception as e:
            print(f"Error loading TF-IDF LR models: {e}")
            return None, None, None, None

    def extract_real_aspects_transformer(self, text, model, tokenizer):
        """Extract real aspects using transformer model"""
        tokens = text.split()
        inputs = tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=128
        )

        word_ids = inputs.word_ids()

        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=2)
        predicted_labels = predictions[0].tolist()

        real_aspects = []
        current_aspect = []

        for word_idx, label_id in zip(word_ids, predicted_labels):
            if word_idx is None:
                continue

            label = self.reverse_real_aspect_label_map.get(label_id, 'O')

            if label == 'B-ASPECT':
                if current_aspect:
                    real_aspects.append(' '.join(current_aspect))
                current_aspect = [tokens[word_idx]]
            elif label == 'I-ASPECT' and current_aspect:
                current_aspect.append(tokens[word_idx])
            elif label == 'O' and current_aspect:
                real_aspects.append(' '.join(current_aspect))
                current_aspect = []

        if current_aspect:
            real_aspects.append(' '.join(current_aspect))

        return real_aspects

    def classify_aspect_transformer(self, sentence, real_aspect, model, tokenizer):
        """Classify real aspect to valid aspect category using transformer"""
        text = f"{sentence} [REAL_ASPECT: {real_aspect}]"

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True
        )

        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=1)

        pred_label = predictions[0].item()
        return self.reverse_aspect_label_map.get(pred_label, 'place')

    def predict_sentiment_transformer(self, sentence_with_aspect, model, tokenizer):
        """Predict sentiment using transformer"""
        inputs = tokenizer(
            sentence_with_aspect,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True
        )

        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=1)

        pred_label = predictions[0].item()
        return self.reverse_sentiment_label_map.get(pred_label, 'neutral')

    def predict_sentiment_tfidf_lr(self, sentence_with_aspect, model, vectorizer):
        """Predict sentiment using TF-IDF + LR"""
        X = vectorizer.transform([sentence_with_aspect])
        prediction = model.predict(X)[0]
        return prediction

    def classify_aspect_tfidf_lr(self, sentence, real_aspect, model, vectorizer):
        """Classify aspect using TF-IDF + LR"""
        text = f"{sentence} [REAL_ASPECT: {real_aspect}]"
        X = vectorizer.transform([text])
        prediction = model.predict(X)[0]
        return prediction

    def perform_absa_inference(self, text, model_name='best'):
        """Perform complete ABSA inference pipeline"""
        results = []

        try:
            if model_name == 'tfidf_lr':
                # Load TF-IDF LR models
                task2_model, task2_vectorizer, task3_model, task3_vectorizer = self.load_tfidf_lr_models()

                if task2_model is None:
                    return f"[Error] Could not load TF-IDF LR models"

                # Task 1: Extract real aspects (rule-based)
                real_aspects = self.extract_real_aspects_rule_based(text)

                if not real_aspects:
                    return f"[TF-IDF LR] No aspects found in: {text}"

                # Process each real aspect
                for real_aspect in real_aspects:
                    # Task 2: Classify to valid aspect
                    valid_aspect = self.classify_aspect_tfidf_lr(text, real_aspect, task2_model, task2_vectorizer)

                    # Task 3: Predict sentiment
                    sentence_with_aspect = f"{text} [ASPECT: {valid_aspect}]"
                    sentiment = self.predict_sentiment_tfidf_lr(sentence_with_aspect, task3_model, task3_vectorizer)

                    results.append({
                        'real_aspect': real_aspect,
                        'aspect_category': valid_aspect,
                        'sentiment': sentiment
                    })

            else:
                # Transformer-based models
                if model_name == 'best':
                    model_name = 'electra'  # Default to electra as best

                # Load models
                task1_model, task1_tokenizer = self.load_transformer_model(model_name, 'task1')
                task2_model, task2_tokenizer = self.load_transformer_model(model_name, 'task2')
                task3_model, task3_tokenizer = self.load_transformer_model(model_name, 'task3')

                # Task 1: Extract real aspects
                real_aspects = self.extract_real_aspects_transformer(text, task1_model, task1_tokenizer)

                if not real_aspects:
                    # Fallback to rule-based
                    real_aspects = self.extract_real_aspects_rule_based(text)

                if not real_aspects:
                    return f"[{model_name.upper()}] No aspects found in: {text}"

                # Process each real aspect
                for real_aspect in real_aspects:
                    # Task 2: Classify to valid aspect
                    valid_aspect = self.classify_aspect_transformer(text, real_aspect, task2_model, task2_tokenizer)

                    # Task 3: Predict sentiment
                    sentence_with_aspect = f"{text} [ASPECT: {valid_aspect}]"
                    sentiment = self.predict_sentiment_transformer(sentence_with_aspect, task3_model, task3_tokenizer)

                    results.append({
                        'real_aspect': real_aspect,
                        'aspect_category': valid_aspect,
                        'sentiment': sentiment
                    })

        except Exception as e:
            return f"[Error] ABSA inference failed: {str(e)}"

        # Format results
        if results:
            formatted_results = f"[{model_name.upper()}] ABSA Analysis:\n"
            for result in results:
                formatted_results += f"  • {result['real_aspect']} ({result['aspect_category']}) → {result['sentiment']}\n"
            return formatted_results.strip()
        else:
            return f"[{model_name.upper()}] No analysis results"

# Model constants
LIST_MODEL_TRANSLATE = [
    'helsinki',
    'nllb',
    'mbart',
    'lstm'
]

LIST_MODEL_ABSA = [
    'bert',
    'deberta',
    'roberta',
    'electra',
    'tfidf_lr',
    'best'
]

LIST_MODEL_SUMMARIZATION = [
    'first_sentence',
    'textrank',
    'bart_zero_shot',
    'bart_full',
    'bart_sampled',
    'pegasus_zero_shot',
    'pegasus_full',
    'pegasus_sampled',
    'best'
]

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_banner():
    """Display a stylish banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║      ██╗███╗   ██╗██████╗  ██████╗ ████████╗██████╗     ║
    ║      ██║████╗  ██║██╔══██╗██╔═══██╗╚══██╔══╝██╔══██╗    ║
    ║      ██║██╔██╗ ██║██║  ██║██║   ██║   ██║   ██████╔╝    ║
    ║      ██║██║╚██╗██║██║  ██║██║   ██║   ██║   ██╔══██╗    ║
    ║      ██║██║ ╚████║██████╔╝╚██████╔╝   ██║   ██║  ██║    ║
    ║      ╚═╝╚═╝  ╚═══╝╚═════╝  ╚═════╝    ╚═╝   ╚═╝  ╚═╝    ║
    ║                                                          ║
    ║      ████████╗██████╗ ██╗██████╗ ███████╗██╗            ║
    ║      ╚══██╔══╝██╔══██╗██║██╔══██╗██╔════╝██║            ║
    ║         ██║   ██████╔╝██║██████╔╝███████╗██║            ║
    ║         ██║   ██╔══██╗██║██╔═══╝ ╚════██║██║            ║
    ║         ██║   ██║  ██║██║██║     ███████║██║            ║
    ║         ╚═╝   ╚═╝  ╚═╝╚═╝╚═╝     ╚══════╝╚═╝            ║
    ║                                                          ║
    ║          IndoTripSight NLP Pipeline System               ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """
    print(banner)

def animate_typing(text, delay=0.03):
    """Animate text appearing character by character."""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

def show_progress_bar(message, duration=2, width=50):
    """Display an animated progress bar."""
    print(f"\n{message}")
    print("[" + " " * width + "]", end="")
    sys.stdout.flush()
    
    for i in range(width + 1):
        progress = i / width
        filled = int(width * progress)
        bar = "█" * filled + "░" * (width - filled)
        percentage = int(progress * 100)
        
        sys.stdout.write(f"\r[{bar}] {percentage}%")
        sys.stdout.flush()
        time.sleep(duration / width)
    
    print(f"\n[COMPLETE] {message} - Complete!")

def animate_spinner(message, duration=2):
    """Display an animated spinner."""
    spinner_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    end_time = time.time() + duration
    
    while time.time() < end_time:
        for char in spinner_chars:
            if time.time() >= end_time:
                break
            sys.stdout.write(f'\r{message} {char}')
            sys.stdout.flush()
            time.sleep(0.1)
    
    sys.stdout.write(f'\r{message} [DONE]\n')
    sys.stdout.flush()

def show_loading_dots(message, duration=2, dot_count=3):
    """Display loading dots animation."""
    end_time = time.time() + duration
    
    while time.time() < end_time:
        for i in range(dot_count + 1):
            if time.time() >= end_time:
                break
            dots = "." * i + " " * (dot_count - i)
            sys.stdout.write(f'\r{message}{dots}')
            sys.stdout.flush()
            time.sleep(0.3)
    
    sys.stdout.write(f'\r{message}{"." * dot_count} [DONE]\n')
    sys.stdout.flush()

def animate_glitch_text(text, iterations=5):
    """Create a glitch effect for text."""
    glitch_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?~`"
    original = text
    for _ in range(iterations):
        glitched = ''.join(random.choice(glitch_chars) if random.random() < 0.3 else c for c in original)
        sys.stdout.write(f'\r{glitched}')
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write(f'\r{original}\n')
    sys.stdout.flush()

def show_wave_animation(message="Welcome", cycles=3):
    """Display a wave animation."""
    wave_frames = [
        "     ┌───────────────────────────┐",
        "    ╔═══╗                        ",
        "   ╔═══╝  IndoTripSight NLP      ",
        "  ╔═══╗   Pipeline System        ",
        " ╔═══╝                           ",
        "╔═══╗                            "
    ]
    
    for cycle in range(cycles):
        for frame in wave_frames:
            clear_screen()
            print_banner()
            print("\n")
            print(" " * 20 + frame)
            print("\n" + " " * 25 + message)
            time.sleep(0.2)
    
    clear_screen()
    print_banner()

def celebrate_completion():
    """Display celebration animation."""
    messages = [
        "*", "#", "=", "+", "-", "~"
    ]
    
    for _ in range(3):
        celebration = " ".join(random.choice(messages) for _ in range(15))
        print("\r" + celebration, end="")
        time.sleep(0.3)
    
    print("\n")

def get_user_choice(options_list, prompt):
    """Get user choice by number from a list of options."""
    while True:
        try:
            choice = int(input(prompt).strip())
            if 1 <= choice <= len(options_list):
                return options_list[choice - 1]
            else:
                print(f"Please enter a number between 1 and {len(options_list)}")
        except ValueError:
            print("Please enter a valid number")


class IndoTripSight:
    def __init__(self):
        self.reviews = []
        self.translated_text = None
        self.absa_processor = ABSA()
        self.translator = Translation()
        self.summarizer = Summarization()
        self.translation_models_loaded = {}
        
    def load_translation_models(self, models_to_load=None):
        """Load specified translation models or all available models."""
        if models_to_load is None:
            models_to_load = ['helsinki', 'nllb', 'mbart', 'lstm']
            
        print("Loading translation models...")
        
        if 'helsinki' in models_to_load:
            try:
                self.translator.load_helsinki_model("/content/drive/MyDrive/Model/fine_tuned_helsinki_tourism2")
                self.translation_models_loaded['helsinki'] = True
                print("✓ Helsinki model loaded")
            except Exception as e:
                print(f"✗ Helsinki model failed: {e}")
                self.translation_models_loaded['helsinki'] = False
                
        if 'nllb' in models_to_load:
            try:
                self.translator.load_nllb_model("/content/drive/MyDrive/Model/fine_tuned_nllb_tourism_essential2")
                self.translation_models_loaded['nllb'] = True
                print("✓ NLLB model loaded")
            except Exception as e:
                print(f"✗ NLLB model failed: {e}")
                self.translation_models_loaded['nllb'] = False
                
        if 'mbart' in models_to_load:
            try:
                self.translator.load_mbart_model("/content/drive/MyDrive/Model/mbart_fine_tuned_essential2")
                self.translation_models_loaded['mbart'] = True
                print("✓ mBART model loaded")
            except Exception as e:
                print(f"✗ mBART model failed: {e}")
                self.translation_models_loaded['mbart'] = False
                
        if 'lstm' in models_to_load:
            try:
                self.translator.load_lstm_model("/content/drive/MyDrive/Model/LSTM2.pt")
                self.translation_models_loaded['lstm'] = True
                print("✓ LSTM model loaded")
            except Exception as e:
                print(f"✗ LSTM model failed: {e}")
                self.translation_models_loaded['lstm'] = False
                
        print(f"Translation models loaded: {list(k for k, v in self.translation_models_loaded.items() if v)}")

    def add_review(self, review):
        self.reviews.append(review)

    def translate_review(self, model='nllb'):
        if not self.reviews:
            return "No reviews to translate"
        review_text = self.reviews[-1]

        try:
            translator_type = model
            
            # Check if model is loaded
            if translator_type not in self.translation_models_loaded or not self.translation_models_loaded[translator_type]:
                self.translated_text = review_text
                return f"[{model.upper()}] Model not loaded. Using original text: {review_text}"
            
            # Perform actual translation
            if translator_type == 'lstm':
                translated = self.translator.translate_text(review_text, 'lstm')
            elif translator_type == 'helsinki':
                translated = self.translator.translate_text(review_text, 'helsinki')
            elif translator_type == 'nllb':
                translated = self.translator.translate_text(review_text, 'nllb', 
                                                          src_lang="ind_Latn", tgt_lang="eng_Latn")
            elif translator_type == 'mbart':
                translated = self.translator.translate_text(review_text, 'mbart',
                                                          src_lang="id_ID", tgt_lang="en_XX")
            else:
                translated = review_text
            
            # Store translated text for ABSA
            self.translated_text = translated if translated else review_text
            
            return f"[{model.upper()}] Translated: {self.translated_text}"
            
        except Exception as e:
            self.translated_text = review_text
            return f"[{model.upper()}] Translation failed: {str(e)}. Using original: {review_text}"

    def get_translation_model_status(self):
        """Get status of loaded translation models."""
        return {
            'loaded_models': [k for k, v in self.translation_models_loaded.items() if v],
            'failed_models': [k for k, v in self.translation_models_loaded.items() if not v],
            'total_loaded': sum(self.translation_models_loaded.values()),
            'available_models': self.translator.get_available_models() if hasattr(self.translator, 'get_available_models') else []
        }
    
    def absa(self, model='best'):
        if self.translated_text is None:
             self.translate_review()
        
        translated_review = self.translated_text

        # Perform actual ABSA inference
        absa_result = self.absa_processor.perform_absa_inference(translated_review, model)

        return absa_result

    def summarization(self, model='best'):
        """Generate summary of the translated review using specified model.
        
        Args:
            model (str): Summarization method to use
            
        Returns:
            str: Generated summary
        """
        # Use the already translated text if available, otherwise translate first
        if self.translated_text is None:
            self.translate_review('nllb')  # Use default translation model
        
        translated_review = self.translated_text
        
        # Generate summary using the Summarization module
        try:
            summary = self.summarizer.summarize(translated_review, method=model)
            return summary
        except Exception as e:
            print(f"Error in summarization: {e}")
            return f"Error generating summary: {str(e)}"

if __name__ == "__main__":
    clear_screen()
    
    # Startup animation with wave effect
    show_wave_animation("Initializing...", cycles=3)
    
    print_banner()
    time.sleep(0.5)
    
    indo_trip_sight = IndoTripSight()
    
    # Initialize translation models
    print("\n" + "═" * 60)
    animate_typing("Initializing Translation Models...")
    print("═" * 60)
    
    models_to_load = ['nllb', 'helsinki', 'mbart', 'lstm']
    indo_trip_sight.load_translation_models(models_to_load)
    
    # Show model status
    status = indo_trip_sight.get_translation_model_status()
    print(f"\n\u2713 Translation models ready: {len(status['loaded_models'])}/{len(models_to_load)}")
    if status['loaded_models']:
        print(f"   Loaded: {', '.join(status['loaded_models'])}")
    if status['failed_models']:
        print(f"   Failed: {', '.join(status['failed_models'])}")

    print("\n" + "═" * 60)
    animate_typing("IndoTripSight NLP Pipeline System Ready")
    print("═" * 60)

    # Get user input for text
    print("\n")
    text = input("Enter the review text: ").strip()
    if text:
        indo_trip_sight.add_review(text)
        animate_spinner("Adding review to pipeline", 1.0)
    else:
        print("[ERROR] No text entered. Exiting...")
        sys.exit(1)

    print("\n" + "═" * 60)

    # Translation Model Selection
    print("\nAvailable Translation Models:")
    for i, model in enumerate(LIST_MODEL_TRANSLATE, 1):
        print(f"   {i}. {model}")

    translate_model = get_user_choice(
        LIST_MODEL_TRANSLATE,
        f"\nChoose translation model (1-{len(LIST_MODEL_TRANSLATE)}): "
    )

    # ABSA Model Selection
    print("\nAvailable ABSA Models:")
    for i, model in enumerate(LIST_MODEL_ABSA, 1):
        print(f"   {i}. {model}")

    absa_model = get_user_choice(
        LIST_MODEL_ABSA,
        f"\nChoose ABSA model (1-{len(LIST_MODEL_ABSA)}): "
    )

    # Summarization Model Selection
    print("\nAvailable Summarization Models:")
    for i, model in enumerate(LIST_MODEL_SUMMARIZATION, 1):
        print(f"   {i}. {model}")

    summarization_model = get_user_choice(
        LIST_MODEL_SUMMARIZATION,
        f"\nChoose summarization model (1-{len(LIST_MODEL_SUMMARIZATION)}): "
    )

    print("\n" + "═" * 60)
    print("Configuration Summary:")
    print(f"   Review: {text}")
    print(f"   Translation: {translate_model}")
    print(f"   ABSA: {absa_model}")
    print(f"   Summarization: {summarization_model}")
    print("═" * 60)
    
    # Show selected model info
    selected_translator_model = translate_model
    
    if selected_translator_model in status['loaded_models']:
        print(f"\n\u2713 Using loaded {selected_translator_model.upper()} model for translation")
    else:
        print(f"\n\u26A0 {selected_translator_model.upper()} model not loaded, will use fallback")

    # Process translation with progress bar
    print("\n")
    show_progress_bar("Translating text", 2.5, width=40)
    translated = indo_trip_sight.translate_review(translate_model)
    print(f"[RESULT] Translation Result: {translated}")

    # Process ABSA with spinner
    print("\n")
    animate_spinner("Analyzing aspects and sentiment", 2.5)
    absa_result = indo_trip_sight.absa(absa_model)
    print(f"[RESULT] ABSA Result: {absa_result}")

    # Process summarization with loading dots
    print("\n")
    show_loading_dots("Generating summary", 2.5, dot_count=3)
    summary = indo_trip_sight.summarization(summarization_model)
    print(f"[RESULT] Summary Result: {summary}")

    # Completion animation with celebration
    print("\n")
    celebrate_completion()
    print("═" * 60)
    animate_glitch_text("Processing complete! Thank you for using IndoTripSight NLP Pipeline!", iterations=5)
    print("═" * 60)