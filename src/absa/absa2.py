import os
import shutil
from google.colab import files
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification, Trainer, TrainingArguments, DataCollatorWithPadding, DataCollatorForTokenClassification, TrainerCallback
from datasets import Dataset
import torch
import nltk
nltk.download('brown')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

os.environ["WANDB_DISABLED"] = "true"

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

class LossHistoryCallback(TrainerCallback):
    def __init__(self):
        self.train_losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and 'loss' in logs:
            self.train_losses.append(logs['loss'])

class ABSA:
    def __init__(self):
        self.results = {}
        self.train_df = None
        self.test_df = None
        self.valid_aspects = VALID_ASPECTS
        self.aspect_label_map = {asp: idx for idx, asp in enumerate(self.valid_aspects)}
        self.reverse_aspect_label_map = {idx: asp for idx, asp in enumerate(self.valid_aspects)}
        self.sentiment_label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        self.reverse_sentiment_label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
        self.real_aspect_label_map = {'O': 0, 'B-ASPECT': 1, 'I-ASPECT': 2}
        self.reverse_real_aspect_label_map = {0: 'O', 1: 'B-ASPECT', 2: 'I-ASPECT'}
        self.loss_histories = {}
        self.task1_results = {}
        self.task2_results = {} 
        self.task3_results = {} 

        os.makedirs('models', exist_ok=True)
        os.environ['WANDB_DISABLED'] = 'true'

    def prepare_task1_data(self, df):
        """Task 1: Prepare data for real aspect extraction (NER-like task)"""
        extraction_data = []

        for idx, row in df.iterrows():
            sentence = str(row['english_translation'])

            if pd.isna(row['real_aspect']):
                continue

            real_aspects = [asp.strip() for asp in str(row['real_aspect']).split(',')]

            # Tokenize sentence
            tokens = sentence.lower().split()
            labels = ['O'] * len(tokens)

            # Tag real aspect words with BIO scheme
            for real_aspect in real_aspects:
                real_aspect_tokens = real_aspect.lower().split()

                # Find occurrences of real_aspect in sentence
                for i in range(len(tokens) - len(real_aspect_tokens) + 1):
                    if tokens[i:i+len(real_aspect_tokens)] == real_aspect_tokens:
                        labels[i] = 'B-ASPECT'
                        for j in range(1, len(real_aspect_tokens)):
                            if i + j < len(labels):
                                labels[i + j] = 'I-ASPECT'

            extraction_data.append({
                'sentence': sentence,
                'tokens': tokens,
                'labels': labels,
                'real_aspects': real_aspects
            })

        return pd.DataFrame(extraction_data)

    def prepare_task2_data(self, df):
        """Task 2: Prepare data for aspect classification (real_aspect -> valid aspect)"""
        classification_data = []

        for idx, row in df.iterrows():
            sentence = str(row['english_translation'])

            if pd.isna(row['real_aspect']) or pd.isna(row['aspects']):
                continue

            real_aspects = [asp.strip() for asp in str(row['real_aspect']).split(',')]
            valid_aspects = [asp.strip() for asp in str(row['aspects']).split(',')]

            if len(real_aspects) == len(valid_aspects):
                for real_asp, valid_asp in zip(real_aspects, valid_aspects):
                    classification_data.append({
                        'sentence': sentence,
                        'real_aspect': real_asp,
                        'aspect_category': valid_asp,
                        'text': f"{sentence} [REAL_ASPECT: {real_asp}]"
                    })

        return pd.DataFrame(classification_data)

    def prepare_task3_data(self, df):
        """Task 3: Prepare data for sentiment classification"""
        sentiment_data = []

        for idx, row in df.iterrows():
            sentence = str(row['english_translation'])

            if pd.isna(row['aspects']) or pd.isna(row['sentiment']):
                continue

            aspects = [asp.strip() for asp in str(row['aspects']).split(',')]
            sentiments = [sent.strip() for sent in str(row['sentiment']).split(',')]

            if len(aspects) == len(sentiments):
                for aspect, sentiment in zip(aspects, sentiments):
                    sentiment_data.append({
                        'sentence': sentence,
                        'aspect': aspect,
                        'polarity': sentiment,
                        'sentence_with_aspect': f"{sentence} [ASPECT: {aspect}]"
                    })

        return pd.DataFrame(sentiment_data)

    def split_data(self, df, test_size=0.2, random_state=42):
        """Split data and prepare datasets for all 3 tasks"""
        # Split original dataframe
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state
        )

        # Prepare datasets for all tasks
        self.train_task1_df = self.prepare_task1_data(train_df)
        self.test_task1_df = self.prepare_task1_data(test_df)

        self.train_task2_df = self.prepare_task2_data(train_df)
        self.test_task2_df = self.prepare_task2_data(test_df)

        self.train_task3_df = self.prepare_task3_data(train_df)
        self.test_task3_df = self.prepare_task3_data(test_df)

        # Store original dataframes
        self.train_df = train_df
        self.test_df = test_df

        print(f"\n{'='*80}")
        print("DATA SPLIT SUMMARY")
        print('='*80)
        print(f"Train sentences: {len(train_df)}")
        print(f"Test sentences: {len(test_df)}")
        print(f"\nTask 1 - Real Aspect Extraction:")
        print(f"  Train: {len(self.train_task1_df)} sentences")
        print(f"  Test:  {len(self.test_task1_df)} sentences")
        print(f"\nTask 2 - Aspect Classification:")
        print(f"  Train: {len(self.train_task2_df)} real_aspect-category pairs")
        print(f"  Test:  {len(self.test_task2_df)} real_aspect-category pairs")
        print(f"\nTask 3 - Sentiment Classification:")
        print(f"  Train: {len(self.train_task3_df)} aspect-sentiment pairs")
        print(f"  Test:  {len(self.test_task3_df)} aspect-sentiment pairs")
        print('='*80)

        return train_df, test_df

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

    def train_real_aspect_extractor_transformer(self, train_df, model_name, output_dir):
        """Train transformer model for real aspect extraction (Task 1)"""
        if 'roberta' in model_name.lower():
            tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)

        def tokenize_and_align_labels(examples):
            tokenized_inputs = tokenizer(
                examples['tokens'],
                truncation=True,
                is_split_into_words=True,
                max_length=128,
                padding='max_length'
            )

            labels = []
            for i, label in enumerate(examples['labels']):
                word_ids = tokenized_inputs.word_ids(batch_index=i)
                label_ids = []
                previous_word_idx = None

                for word_idx in word_ids:
                    if word_idx is None:
                        label_ids.append(-100)
                    elif word_idx != previous_word_idx:
                        label_ids.append(self.real_aspect_label_map[label[word_idx]])
                    else:
                        label_ids.append(-100)
                    previous_word_idx = word_idx

                labels.append(label_ids)

            tokenized_inputs["labels"] = labels
            return tokenized_inputs

        train_dataset = Dataset.from_dict({
            'tokens': train_df['tokens'].tolist(),
            'labels': train_df['labels'].tolist()
        })

        train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)

        model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=len(self.real_aspect_label_map)
        )

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=16,
            warmup_steps=50,
            weight_decay=0.01,
            logging_steps=10,
            save_strategy="no",
            report_to="none",
            disable_tqdm=False
        )

        data_collator = DataCollatorForTokenClassification(tokenizer)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator
        )

        trainer.train()

        return model, tokenizer

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

    def calculate_task1_metrics(self, extracted, true):
        """Calculate metrics for Task 1 (Real Aspect Extraction)"""
        total_extracted = 0
        total_true = 0
        total_correct = 0

        for ext_list, true_list in zip(extracted, true):
            ext_set = set([asp.lower().strip() for asp in ext_list])
            true_set = set([asp.lower().strip() for asp in true_list])

            total_extracted += len(ext_set)
            total_true += len(true_set)
            total_correct += len(ext_set & true_set)

        precision = total_correct / total_extracted if total_extracted > 0 else 0
        recall = total_correct / total_true if total_true > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return precision, recall, f1

    def train_aspect_classifier_transformer(self, train_df, test_df, model_name, output_dir, method_name):
        """Train transformer model for aspect classification (Task 2)"""
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        train_data = Dataset.from_pandas(
            train_df[['text', 'aspect_category']].rename(
                columns={'text': 'text', 'aspect_category': 'label'}
            )
        )
        test_data = Dataset.from_pandas(
            test_df[['text', 'aspect_category']].rename(
                columns={'text': 'text', 'aspect_category': 'label'}
            )
        )

        # Map aspect categories to labels
        train_data = train_data.map(lambda x: {'label': self.aspect_label_map.get(x['label'], 0)})
        test_data = test_data.map(lambda x: {'label': self.aspect_label_map.get(x['label'], 0)})

        def tokenize_function(examples):
            return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)

        train_dataset = train_data.map(tokenize_function, batched=True)
        test_dataset = test_data.map(tokenize_function, batched=True)

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(self.valid_aspects)
        )

        loss_callback = LossHistoryCallback()

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            warmup_steps=100,
            weight_decay=0.01,
            logging_steps=10,
            save_strategy="no",
            report_to="none",
            disable_tqdm=False
        )

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            callbacks=[loss_callback]
        )

        trainer.train()

        # Store loss history
        self.loss_histories[f"{method_name}_task2"] = loss_callback.train_losses

        # Get predictions
        predictions_output = trainer.predict(test_dataset)
        predictions = np.argmax(predictions_output.predictions, axis=1)
        predictions = [self.reverse_aspect_label_map.get(pred, 'place') for pred in predictions]

        return model, tokenizer, predictions

    def classify_aspect_rule_based(self, sentence, real_aspect):
        """Rule-based aspect classification"""
        # Simple matching - find closest valid aspect
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

    def train_sentiment_classifier_transformer(self, train_df, test_df, model_name, output_dir, method_name):
        """Train transformer model for sentiment classification (Task 3)"""
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        train_data = Dataset.from_pandas(
            train_df[['sentence_with_aspect', 'polarity']].rename(
                columns={'sentence_with_aspect': 'text', 'polarity': 'label'}
            )
        )
        test_data = Dataset.from_pandas(
            test_df[['sentence_with_aspect', 'polarity']].rename(
                columns={'sentence_with_aspect': 'text', 'polarity': 'label'}
            )
        )

        train_data = train_data.map(lambda x: {'label': self.sentiment_label_map[x['label']]})
        test_data = test_data.map(lambda x: {'label': self.sentiment_label_map[x['label']]})

        def tokenize_function(examples):
            return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)

        train_dataset = train_data.map(tokenize_function, batched=True)
        test_dataset = test_data.map(tokenize_function, batched=True)

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=3
        )

        loss_callback = LossHistoryCallback()

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            warmup_steps=100,
            weight_decay=0.01,
            logging_steps=10,
            save_strategy="no",
            report_to="none",
            disable_tqdm=False
        )

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            callbacks=[loss_callback]
        )

        trainer.train()

        # Store loss history
        self.loss_histories[f"{method_name}_task3"] = loss_callback.train_losses

        # Get predictions
        predictions_output = trainer.predict(test_dataset)
        predictions = np.argmax(predictions_output.predictions, axis=1)
        predictions = [self.reverse_sentiment_label_map[pred] for pred in predictions]

        return model, tokenizer, predictions

    def method1_vader(self, train_df, test_df):
        """VADER: Rule-based for all 3 tasks"""
        start_time = time.time()
        analyzer = SentimentIntensityAnalyzer()

        # Task 1: Extract real aspects
        extracted_real_aspects = []
        true_real_aspects = []

        for _, row in test_df.iterrows():
            sentence = str(row['english_translation'])
            real_aspects = self.extract_real_aspects_rule_based(sentence)
            extracted_real_aspects.append(real_aspects)

            if not pd.isna(row['real_aspect']):
                true_real_aspects.append([asp.strip() for asp in str(row['real_aspect']).split(',')])
            else:
                true_real_aspects.append([])

        self.task1_results['VADER'] = {
            'extracted': extracted_real_aspects,
            'true': true_real_aspects
        }

        # Task 2: Classify aspects
        aspect_predictions = []
        aspect_true_labels = []

        for _, row in self.test_task2_df.iterrows():
            sentence = str(row['sentence'])
            real_aspect = str(row['real_aspect'])
            pred_aspect = self.classify_aspect_rule_based(sentence, real_aspect)
            aspect_predictions.append(pred_aspect)
            aspect_true_labels.append(row['aspect_category'])

        self.task2_results['VADER'] = {
            'predicted': aspect_predictions,
            'true': aspect_true_labels
        }

        # Task 3: Predict sentiment
        sentiment_predictions = []
        sentiment_true_labels = []

        for _, row in self.test_task3_df.iterrows():
            sentence = str(row['sentence_with_aspect'])
            scores = analyzer.polarity_scores(sentence)

            if scores['compound'] >= 0.05:
                pred = 'positive'
            elif scores['compound'] <= -0.05:
                pred = 'negative'
            else:
                pred = 'neutral'

            sentiment_predictions.append(pred)
            sentiment_true_labels.append(row['polarity'])

        self.task3_results['VADER'] = {
            'predicted': sentiment_predictions,
            'true': sentiment_true_labels
        }

        training_time = time.time() - start_time
        return sentiment_predictions, sentiment_true_labels, training_time

    def method2_textblob(self, train_df, test_df):
        """TextBlob: Rule-based for all 3 tasks"""
        start_time = time.time()

        # Task 1: Extract real aspects
        extracted_real_aspects = []
        true_real_aspects = []

        for _, row in test_df.iterrows():
            sentence = str(row['english_translation'])
            real_aspects = self.extract_real_aspects_rule_based(sentence)
            extracted_real_aspects.append(real_aspects)

            if not pd.isna(row['real_aspect']):
                true_real_aspects.append([asp.strip() for asp in str(row['real_aspect']).split(',')])
            else:
                true_real_aspects.append([])

        self.task1_results['TextBlob'] = {
            'extracted': extracted_real_aspects,
            'true': true_real_aspects
        }

        # Task 2: Classify aspects
        aspect_predictions = []
        aspect_true_labels = []

        for _, row in self.test_task2_df.iterrows():
            sentence = str(row['sentence'])
            real_aspect = str(row['real_aspect'])
            pred_aspect = self.classify_aspect_rule_based(sentence, real_aspect)
            aspect_predictions.append(pred_aspect)
            aspect_true_labels.append(row['aspect_category'])

        self.task2_results['TextBlob'] = {
            'predicted': aspect_predictions,
            'true': aspect_true_labels
        }

        # Task 3: Predict sentiment
        sentiment_predictions = []
        sentiment_true_labels = []

        for _, row in self.test_task3_df.iterrows():
            sentence = str(row['sentence_with_aspect'])
            blob = TextBlob(sentence)
            polarity = blob.sentiment.polarity

            if polarity > 0.1:
                pred = 'positive'
            elif polarity < -0.1:
                pred = 'negative'
            else:
                pred = 'neutral'

            sentiment_predictions.append(pred)
            sentiment_true_labels.append(row['polarity'])

        self.task3_results['TextBlob'] = {
            'predicted': sentiment_predictions,
            'true': sentiment_true_labels
        }

        training_time = time.time() - start_time
        return sentiment_predictions, sentiment_true_labels, training_time

    def method3_tfidf_lr(self, train_df, test_df):
        """TF-IDF + LR: Machine learning for all 3 tasks"""
        start_time = time.time()

        # Task 1: Extract real aspects (rule-based)
        extracted_real_aspects = []
        true_real_aspects = []

        for _, row in test_df.iterrows():
            sentence = str(row['english_translation'])
            real_aspects = self.extract_real_aspects_rule_based(sentence)
            extracted_real_aspects.append(real_aspects)

            if not pd.isna(row['real_aspect']):
                true_real_aspects.append([asp.strip() for asp in str(row['real_aspect']).split(',')])
            else:
                true_real_aspects.append([])

        self.task1_results['TF-IDF + LR'] = {
            'extracted': extracted_real_aspects,
            'true': true_real_aspects
        }

        # Task 2: Classify aspects using TF-IDF + LR
        X_train_task2 = self.train_task2_df['text'].values
        y_train_task2 = self.train_task2_df['aspect_category'].values
        X_test_task2 = self.test_task2_df['text'].values
        y_test_task2 = self.test_task2_df['aspect_category'].values

        vectorizer_task2 = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X_train_vec_task2 = vectorizer_task2.fit_transform(X_train_task2)
        X_test_vec_task2 = vectorizer_task2.transform(X_test_task2)

        model_task2 = LogisticRegression(max_iter=1000, random_state=42)
        model_task2.fit(X_train_vec_task2, y_train_task2)

        aspect_predictions = model_task2.predict(X_test_vec_task2).tolist()

        self.task2_results['TF-IDF + LR'] = {
            'predicted': aspect_predictions,
            'true': y_test_task2.tolist()
        }

        # Task 3: Predict sentiment using TF-IDF + LR
        X_train_task3 = self.train_task3_df['sentence_with_aspect'].values
        y_train_task3 = self.train_task3_df['polarity'].values
        X_test_task3 = self.test_task3_df['sentence_with_aspect'].values
        y_test_task3 = self.test_task3_df['polarity'].values

        vectorizer_task3 = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X_train_vec_task3 = vectorizer_task3.fit_transform(X_train_task3)
        X_test_vec_task3 = vectorizer_task3.transform(X_test_task3)

        model_task3 = LogisticRegression(max_iter=1000, random_state=42)
        model_task3.fit(X_train_vec_task3, y_train_task3)

        sentiment_predictions = model_task3.predict(X_test_vec_task3).tolist()

        # Save models
        joblib.dump(model_task2, 'models/tfidf_lr_task2_model.pkl')
        joblib.dump(vectorizer_task2, 'models/tfidf_lr_task2_vectorizer.pkl')
        joblib.dump(model_task3, 'models/tfidf_lr_task3_model.pkl')
        joblib.dump(vectorizer_task3, 'models/tfidf_lr_task3_vectorizer.pkl')

        self.task3_results['TF-IDF + LR'] = {
            'predicted': sentiment_predictions,
            'true': y_test_task3.tolist()
        }

        training_time = time.time() - start_time
        return sentiment_predictions, y_test_task3.tolist(), training_time

    def method4_bert(self, train_df, test_df):
        """BERT: Transformer for all 3 tasks"""
        try:
            start_time = time.time()
            model_name = "distilbert-base-uncased"

            # Task 1: Train real aspect extractor
            task1_model, task1_tokenizer = self.train_real_aspect_extractor_transformer(
                self.train_task1_df,
                model_name,
                './results_bert_task1'
            )
            task1_model.save_pretrained('models/bert_task1')
            task1_tokenizer.save_pretrained('models/bert_task1')

            # Task 1: Extract real aspects
            extracted_real_aspects = []
            true_real_aspects = []

            for _, row in test_df.iterrows():
                sentence = str(row['english_translation'])
                real_aspects = self.extract_real_aspects_transformer(sentence, task1_model, task1_tokenizer)
                extracted_real_aspects.append(real_aspects)

                if not pd.isna(row['real_aspect']):
                    true_real_aspects.append([asp.strip() for asp in str(row['real_aspect']).split(',')])
                else:
                    true_real_aspects.append([])

            self.task1_results['BERT'] = {
                'extracted': extracted_real_aspects,
                'true': true_real_aspects
            }

            # Task 2: Train aspect classifier
            task2_model, task2_tokenizer, aspect_predictions = self.train_aspect_classifier_transformer(
                self.train_task2_df,
                self.test_task2_df,
                model_name,
                './results_bert_task2',
                'BERT'
            )
            task2_model.save_pretrained('models/bert_task2')
            task2_tokenizer.save_pretrained('models/bert_task2')

            self.task2_results['BERT'] = {
                'predicted': aspect_predictions,
                'true': self.test_task2_df['aspect_category'].tolist()
            }

            # Task 3: Train sentiment classifier
            task3_model, task3_tokenizer, sentiment_predictions = self.train_sentiment_classifier_transformer(
                self.train_task3_df,
                self.test_task3_df,
                model_name,
                './results_bert_task3',
                'BERT'
            )
            task3_model.save_pretrained('models/bert_task3')
            task3_tokenizer.save_pretrained('models/bert_task3')

            self.task3_results['BERT'] = {
                'predicted': sentiment_predictions,
                'true': self.test_task3_df['polarity'].tolist()
            }

            training_time = time.time() - start_time
            return sentiment_predictions, self.test_task3_df['polarity'].tolist(), training_time

        except Exception as e:
            print(f"Error in BERT method: {e}")
            import traceback
            traceback.print_exc()
            return [], [], 0

    def method5_roberta(self, train_df, test_df):
        """RoBERTa: Transformer for all 3 tasks"""
        try:
            start_time = time.time()
            model_name = "roberta-base"

            # Task 1: Train real aspect extractor
            task1_model, task1_tokenizer = self.train_real_aspect_extractor_transformer(
                self.train_task1_df,
                model_name,
                './results_roberta_task1'
            )
            task1_model.save_pretrained('models/roberta_task1')
            task1_tokenizer.save_pretrained('models/roberta_task1')

            # Task 1: Extract real aspects
            extracted_real_aspects = []
            true_real_aspects = []

            for _, row in test_df.iterrows():
                sentence = str(row['english_translation'])
                real_aspects = self.extract_real_aspects_transformer(sentence, task1_model, task1_tokenizer)
                extracted_real_aspects.append(real_aspects)

                if not pd.isna(row['real_aspect']):
                    true_real_aspects.append([asp.strip() for asp in str(row['real_aspect']).split(',')])
                else:
                    true_real_aspects.append([])

            self.task1_results['RoBERTa'] = {
                'extracted': extracted_real_aspects,
                'true': true_real_aspects
            }

            # Task 2: Train aspect classifier
            task2_model, task2_tokenizer, aspect_predictions = self.train_aspect_classifier_transformer(
                self.train_task2_df,
                self.test_task2_df,
                model_name,
                './results_roberta_task2',
                'RoBERTa'
            )
            task2_model.save_pretrained('models/roberta_task2')
            task2_tokenizer.save_pretrained('models/roberta_task2')

            self.task2_results['RoBERTa'] = {
                'predicted': aspect_predictions,
                'true': self.test_task2_df['aspect_category'].tolist()
            }

            # Task 3: Train sentiment classifier
            task3_model, task3_tokenizer, sentiment_predictions = self.train_sentiment_classifier_transformer(
                self.train_task3_df,
                self.test_task3_df,
                model_name,
                './results_roberta_task3',
                'RoBERTa'
            )
            task3_model.save_pretrained('models/roberta_task3')
            task3_tokenizer.save_pretrained('models/roberta_task3')

            self.task3_results['RoBERTa'] = {
                'predicted': sentiment_predictions,
                'true': self.test_task3_df['polarity'].tolist()
            }

            training_time = time.time() - start_time
            return sentiment_predictions, self.test_task3_df['polarity'].tolist(), training_time

        except Exception as e:
            print(f"Error in RoBERTa method: {e}")
            import traceback
            traceback.print_exc()
            return [], [], 0

    def method6_electra(self, train_df, test_df):
        """ELECTRA: Transformer for all 3 tasks"""
        try:
            start_time = time.time()
            model_name = "google/electra-small-discriminator"

            # Task 1: Train real aspect extractor
            task1_model, task1_tokenizer = self.train_real_aspect_extractor_transformer(
                self.train_task1_df,
                model_name,
                './results_electra_task1'
            )
            task1_model.save_pretrained('models/electra_task1')
            task1_tokenizer.save_pretrained('models/electra_task1')

            # Task 1: Extract real aspects
            extracted_real_aspects = []
            true_real_aspects = []

            for _, row in test_df.iterrows():
                sentence = str(row['english_translation'])
                real_aspects = self.extract_real_aspects_transformer(sentence, task1_model, task1_tokenizer)
                extracted_real_aspects.append(real_aspects)

                if not pd.isna(row['real_aspect']):
                    true_real_aspects.append([asp.strip() for asp in str(row['real_aspect']).split(',')])
                else:
                    true_real_aspects.append([])

            self.task1_results['ELECTRA'] = {
                'extracted': extracted_real_aspects,
                'true': true_real_aspects
            }

            # Task 2: Train aspect classifier
            task2_model, task2_tokenizer, aspect_predictions = self.train_aspect_classifier_transformer(
                self.train_task2_df,
                self.test_task2_df,
                model_name,
                './results_electra_task2',
                'ELECTRA'
            )
            task2_model.save_pretrained('models/electra_task2')
            task2_tokenizer.save_pretrained('models/electra_task2')

            self.task2_results['ELECTRA'] = {
                'predicted': aspect_predictions,
                'true': self.test_task2_df['aspect_category'].tolist()
            }

            # Task 3: Train sentiment classifier
            task3_model, task3_tokenizer, sentiment_predictions = self.train_sentiment_classifier_transformer(
                self.train_task3_df,
                self.test_task3_df,
                model_name,
                './results_electra_task3',
                'ELECTRA'
            )
            task3_model.save_pretrained('models/electra_task3')
            task3_tokenizer.save_pretrained('models/electra_task3')

            self.task3_results['ELECTRA'] = {
                'predicted': sentiment_predictions,
                'true': self.test_task3_df['polarity'].tolist()
            }

            training_time = time.time() - start_time
            return sentiment_predictions, self.test_task3_df['polarity'].tolist(), training_time

        except Exception as e:
            print(f"Error in ELECTRA method: {e}")
            import traceback
            traceback.print_exc()
            return [], [], 0

    def method7_deberta(self, train_df, test_df):
        """DeBERTa: Transformer for all 3 tasks"""
        try:
            start_time = time.time()
            model_name = "microsoft/deberta-v3-small"

            # Task 1: Train real aspect extractor
            task1_model, task1_tokenizer = self.train_real_aspect_extractor_transformer(
                self.train_task1_df,
                model_name,
                './results_deberta_task1'
            )
            task1_model.save_pretrained('models/deberta_task1')
            task1_tokenizer.save_pretrained('models/deberta_task1')

            # Task 1: Extract real aspects
            extracted_real_aspects = []
            true_real_aspects = []

            for _, row in test_df.iterrows():
                sentence = str(row['english_translation'])
                real_aspects = self.extract_real_aspects_transformer(sentence, task1_model, task1_tokenizer)
                extracted_real_aspects.append(real_aspects)

                if not pd.isna(row['real_aspect']):
                    true_real_aspects.append([asp.strip() for asp in str(row['real_aspect']).split(',')])
                else:
                    true_real_aspects.append([])

            self.task1_results['DeBERTa'] = {
                'extracted': extracted_real_aspects,
                'true': true_real_aspects
            }

            # Task 2: Train aspect classifier
            task2_model, task2_tokenizer, aspect_predictions = self.train_aspect_classifier_transformer(
                self.train_task2_df,
                self.test_task2_df,
                model_name,
                './results_deberta_task2',
                'DeBERTa'
            )
            task2_model.save_pretrained('models/deberta_task2')
            task2_tokenizer.save_pretrained('models/deberta_task2')

            self.task2_results['DeBERTa'] = {
                'predicted': aspect_predictions,
                'true': self.test_task2_df['aspect_category'].tolist()
            }

            # Task 3: Train sentiment classifier
            task3_model, task3_tokenizer, sentiment_predictions = self.train_sentiment_classifier_transformer(
                self.train_task3_df,
                self.test_task3_df,
                model_name,
                './results_deberta_task3',
                'DeBERTa'
            )
            task3_model.save_pretrained('models/deberta_task3')
            task3_tokenizer.save_pretrained('models/deberta_task3')

            self.task3_results['DeBERTa'] = {
                'predicted': sentiment_predictions,
                'true': self.test_task3_df['polarity'].tolist()
            }

            training_time = time.time() - start_time
            return sentiment_predictions, self.test_task3_df['polarity'].tolist(), training_time

        except Exception as e:
            print(f"Error in DeBERTa method: {e}")
            import traceback
            traceback.print_exc()
            return [], [], 0


    def calculate_task2_metrics(self, predicted, true):
        """Calculate metrics for Task 2 (Aspect Classification)"""
        accuracy = accuracy_score(true, predicted)
        labels = sorted(list(set(true) | set(predicted)))

        precision = precision_score(true, predicted, average='weighted', labels=labels, zero_division=0)
        recall = recall_score(true, predicted, average='weighted', labels=labels, zero_division=0)
        f1 = f1_score(true, predicted, average='weighted', labels=labels, zero_division=0)

        return accuracy, precision, recall, f1

    def evaluate_method(self, y_true, y_pred, method_name, training_time):
        """Evaluate method performance on all 3 tasks"""

        # Task 3: Sentiment Classification Metrics
        accuracy = accuracy_score(y_true, y_pred)
        labels = sorted(list(set(y_true) | set(y_pred)))

        precision = precision_score(y_true, y_pred, average='weighted', labels=labels, zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', labels=labels, zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', labels=labels, zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', labels=labels, zero_division=0)

        print(f"\n[TASK 3: SENTIMENT CLASSIFICATION METRICS]")
        print(f"Accuracy:            {accuracy:.4f}")
        print(f"Precision (Weighted): {precision:.4f}")
        print(f"Recall (Weighted):    {recall:.4f}")
        print(f"F1-Score (Weighted):  {f1:.4f}")
        print(f"F1-Score (Macro):     {f1_macro:.4f}")

        # Task 1: Real Aspect Extraction Metrics
        if method_name in self.task1_results:
            task1_data = self.task1_results[method_name]
            task1_precision, task1_recall, task1_f1 = self.calculate_task1_metrics(
                task1_data['extracted'],
                task1_data['true']
            )

            print(f"\n[TASK 1: REAL ASPECT EXTRACTION METRICS]")
            print(f"Precision: {task1_precision:.4f}")
            print(f"Recall:    {task1_recall:.4f}")
            print(f"F1-Score:  {task1_f1:.4f}")
        else:
            task1_precision = task1_recall = task1_f1 = 0.0

        # Task 2: Aspect Classification Metrics
        if method_name in self.task2_results:
            task2_data = self.task2_results[method_name]
            task2_accuracy, task2_precision, task2_recall, task2_f1 = self.calculate_task2_metrics(
                task2_data['predicted'],
                task2_data['true']
            )

            print(f"\n[TASK 2: ASPECT CLASSIFICATION METRICS]")
            print(f"Accuracy:  {task2_accuracy:.4f}")
            print(f"Precision: {task2_precision:.4f}")
            print(f"Recall:    {task2_recall:.4f}")
            print(f"F1-Score:  {task2_f1:.4f}")
        else:
            task2_accuracy = task2_precision = task2_recall = task2_f1 = 0.0

        print(f"\n[TRAINING TIME]")
        print(f"Total Time: {training_time:.4f} seconds")
        print('='*80)

        cm = confusion_matrix(y_true, y_pred, labels=labels)

        return {
            'method': method_name,
            # Task 1 metrics
            'task1_precision': task1_precision,
            'task1_recall': task1_recall,
            'task1_f1': task1_f1,
            # Task 2 metrics
            'task2_accuracy': task2_accuracy,
            'task2_precision': task2_precision,
            'task2_recall': task2_recall,
            'task2_f1': task2_f1,
            # Task 3 metrics
            'task3_accuracy': accuracy,
            'task3_precision': precision,
            'task3_recall': recall,
            'task3_f1': f1,
            'task3_f1_macro': f1_macro,
            'training_time': training_time,
            'predictions': y_pred,
            'confusion_matrix': cm,
            'labels': labels
        }

    def compare_all_methods(self):
        """Run all methods and compare results"""
        if self.train_df is None or self.test_df is None:
            raise ValueError("Please split data first using split_data()")

        methods = {
            'VADER': self.method1_vader,
            'TextBlob': self.method2_textblob,
            'TF-IDF + LR': self.method3_tfidf_lr,
            'BERT': self.method4_bert,
            'RoBERTa': self.method5_roberta,
            'ELECTRA': self.method6_electra,
            'DeBERTa': self.method7_deberta,
        }

        all_results = []

        for method_name, method_func in methods.items():
            try:
                print(f"\n{'='*80}")
                print(f"Running {method_name}...")
                print('='*80)

                predictions, true_labels, training_time = method_func(self.train_df, self.test_df)

                if len(predictions) > 0 and len(true_labels) > 0:
                    result = self.evaluate_method(true_labels, predictions, method_name, training_time)
                    all_results.append(result)
                else:
                    print(f"Skipping {method_name} due to errors")

            except Exception as e:
                print(f"\nError in {method_name}: {e}")
                import traceback
                traceback.print_exc()
                continue

        self.results = all_results
        return all_results

    def plot_results(self):
        """Plot results for all 3 tasks"""
        comparison_df = pd.DataFrame(self.results)
        pastel_colors = ['#FFB3BA', '#BAFFC9', '#BAE1FF', '#FFFFBA', '#E0BBE4', '#FFD9B3', '#C9FFE5', '#E6B3FF']

        # 1. TASK 1: REAL ASPECT EXTRACTION METRICS
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Task 1: Real Aspect Extraction Performance', fontsize=16, fontweight='bold')

        task1_metrics = [
            ('task1_precision', 'Precision', axes[0]),
            ('task1_recall', 'Recall', axes[1]),
            ('task1_f1', 'F1-Score', axes[2])
        ]

        for metric, title, ax in task1_metrics:
            bars = ax.bar(comparison_df['method'], comparison_df[metric],
                          color=pastel_colors[:len(comparison_df)])
            ax.set_ylabel(title, fontsize=12)
            ax.set_title(title, fontsize=13, fontweight='bold')
            ax.set_ylim([0, 1])
            ax.tick_params(axis='x', rotation=45, labelsize=10)

            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig('task1_real_aspect_extraction_metrics.png', dpi=300, bbox_inches='tight')
        print("\nSaved: task1_real_aspect_extraction_metrics.png")
        plt.show()

        # 2. TASK 2: ASPECT CLASSIFICATION METRICS
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Task 2: Aspect Classification Performance', fontsize=16, fontweight='bold')

        task2_metrics = [
            ('task2_accuracy', 'Accuracy', axes[0, 0]),
            ('task2_precision', 'Precision', axes[0, 1]),
            ('task2_recall', 'Recall', axes[1, 0]),
            ('task2_f1', 'F1-Score', axes[1, 1])
        ]

        for metric, title, ax in task2_metrics:
            bars = ax.bar(comparison_df['method'], comparison_df[metric],
                          color=pastel_colors[:len(comparison_df)])
            ax.set_ylabel(title, fontsize=12)
            ax.set_title(title, fontsize=13, fontweight='bold')
            ax.set_ylim([0, 1])
            ax.tick_params(axis='x', rotation=45, labelsize=10)

            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig('task2_aspect_classification_metrics.png', dpi=300, bbox_inches='tight')
        print("Saved: task2_aspect_classification_metrics.png")
        plt.show()

        # 3. TASK 3: SENTIMENT CLASSIFICATION METRICS
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Task 3: Sentiment Classification Performance', fontsize=16, fontweight='bold')

        task3_metrics = [
            ('task3_accuracy', 'Accuracy', axes[0, 0]),
            ('task3_precision', 'Precision', axes[0, 1]),
            ('task3_recall', 'Recall', axes[1, 0]),
            ('task3_f1', 'F1-Score', axes[1, 1])
        ]

        for metric, title, ax in task3_metrics:
            bars = ax.bar(comparison_df['method'], comparison_df[metric],
                          color=pastel_colors[:len(comparison_df)])
            ax.set_ylabel(title, fontsize=12)
            ax.set_title(title, fontsize=13, fontweight='bold')
            ax.set_ylim([0, 1])
            ax.tick_params(axis='x', rotation=45, labelsize=10)

            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig('task3_sentiment_classification_metrics.png', dpi=300, bbox_inches='tight')
        print("Saved: task3_sentiment_classification_metrics.png")
        plt.show()

        # 4. TRAINING TIME COMPARISON
        plt.figure(figsize=(12, 6))
        bars = plt.bar(comparison_df['method'], comparison_df['training_time'],
                       color=pastel_colors[:len(comparison_df)])
        plt.ylabel('Training Time (seconds)', fontsize=12)
        plt.title('Training Time Comparison (All 3 Tasks)', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')

        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}s', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig('training_time_comparison.png', dpi=300, bbox_inches='tight')
        print("Saved: training_time_comparison.png")
        plt.show()

        # 5. TRAINING LOSS CURVES
        if self.loss_histories:
            plt.figure(figsize=(14, 8))

            for method_name, losses in self.loss_histories.items():
                if losses:
                    plt.plot(losses, label=method_name, linewidth=2, marker='o', markersize=4)

            plt.xlabel('Training Steps', fontsize=12)
            plt.ylabel('Loss', fontsize=12)
            plt.title('Training Loss Curves (Task 2 & Task 3)', fontsize=14, fontweight='bold')
            plt.legend(fontsize=10, loc='best')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('training_loss_curves.png', dpi=300, bbox_inches='tight')
            print("Saved: training_loss_curves.png")
            plt.show()

        # 6. CONFUSION MATRICES (Task 3)
        for i, result in enumerate(self.results):
            plt.figure(figsize=(8, 6))
            cm = result['confusion_matrix']
            labels = result['labels']

            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='RdYlGn',
                       xticklabels=labels, yticklabels=labels,
                       cbar_kws={'label': 'Normalized Accuracy'})
            plt.ylabel('True Label', fontsize=12)
            plt.xlabel('Predicted Label', fontsize=12)
            plt.title(f'{result["method"]} - Task 3 Confusion Matrix',
                     fontsize=14, fontweight='bold')
            plt.tight_layout()

            filename = f'confusion_matrix_task3_{result["method"].replace(" ", "_").replace("+", "").lower()}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")
            plt.show()

        print("\n[TASK 1: REAL ASPECT EXTRACTION METRICS]")
        task1_df = comparison_df[['method', 'task1_precision', 'task1_recall', 'task1_f1']].copy()
        print(task1_df.to_string(index=False))

        print("\n[TASK 2: ASPECT CLASSIFICATION METRICS]")
        task2_df = comparison_df[['method', 'task2_accuracy', 'task2_precision', 'task2_recall', 'task2_f1']].copy()
        print(task2_df.to_string(index=False))

        print("\n[TASK 3: SENTIMENT CLASSIFICATION METRICS]")
        task3_df = comparison_df[['method', 'task3_accuracy', 'task3_precision', 'task3_recall', 'task3_f1', 'task3_f1_macro']].copy()
        print(task3_df.to_string(index=False))

        print("\n[TRAINING TIME]")
        time_df = comparison_df[['method', 'training_time']].copy()
        time_df['training_time'] = time_df['training_time'].apply(lambda x: f'{x:.4f}s')
        print(time_df.to_string(index=False))
