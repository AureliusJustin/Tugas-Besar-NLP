import os
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
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    Trainer, 
    TrainingArguments,
    DataCollatorWithPadding,
    DataCollatorForTokenClassification,
    TrainerCallback
)
from datasets import Dataset
import torch
import nltk


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
        
        # Sentiment labels
        self.label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        self.reverse_label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
        
        # Aspect extraction labels (BIO tagging)
        self.aspect_label_map = {'O': 0, 'B-ASPECT': 1, 'I-ASPECT': 2}
        self.reverse_aspect_label_map = {0: 'O', 1: 'B-ASPECT', 2: 'I-ASPECT'}
        
        # Storage for results
        self.loss_histories = {}
        self.aspect_extraction_results = {}
        
        os.makedirs('models', exist_ok=True)
        os.environ['WANDB_DISABLED'] = 'true'
    
    def prepare_aspect_extraction_data(self, df):
        extraction_data = []
        
        for idx, row in df.iterrows():
            sentence = str(row['english_translation'])
            
            if pd.isna(row['aspects']):
                continue
            
            aspects = [asp.strip() for asp in str(row['aspects']).split(',')]
            
            # Tokenize sentence (simple word-level tokenization)
            tokens = sentence.lower().split()
            labels = ['O'] * len(tokens)
            
            # Tag aspect words with BIO scheme
            for aspect in aspects:
                aspect_tokens = aspect.lower().split()
                
                # Find aspect in sentence
                for i in range(len(tokens) - len(aspect_tokens) + 1):
                    if tokens[i:i+len(aspect_tokens)] == aspect_tokens:
                        labels[i] = 'B-ASPECT'
                        for j in range(1, len(aspect_tokens)):
                            if i + j < len(labels):
                                labels[i + j] = 'I-ASPECT'
                        break
            
            extraction_data.append({
                'sentence': sentence,
                'tokens': tokens,
                'labels': labels,
                'aspects': aspects
            })
        
        return pd.DataFrame(extraction_data)
    
    def prepare_sentiment_data(self, df):
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
        # Split original dataframe
        train_df, test_df = train_test_split(
            df, 
            test_size=test_size, 
            random_state=random_state
        )
        
        # Prepare datasets for both tasks
        self.train_extraction_df = self.prepare_aspect_extraction_data(train_df)
        self.test_extraction_df = self.prepare_aspect_extraction_data(test_df)
        
        self.train_sentiment_df = self.prepare_sentiment_data(train_df)
        self.test_sentiment_df = self.prepare_sentiment_data(test_df)
        
        # Store original dataframes
        self.train_df = train_df
        self.test_df = test_df
        
        print(f"\n{'='*80}")
        print("DATA SPLIT SUMMARY")
        print('='*80)
        print(f"Train sentences: {len(train_df)}")
        print(f"Test sentences: {len(test_df)}")
        print(f"\nAspect Extraction Dataset:")
        print(f"  Train: {len(self.train_extraction_df)} sentences")
        print(f"  Test:  {len(self.test_extraction_df)} sentences")
        print(f"\nSentiment Classification Dataset:")
        print(f"  Train: {len(self.train_sentiment_df)} aspect-sentiment pairs")
        print(f"  Test:  {len(self.test_sentiment_df)} aspect-sentiment pairs")
        print('='*80)
        
        return train_df, test_df
    
    def extract_aspects_rule_based(self, text):
        blob = TextBlob(text)
        aspects = []
        
        # Extract noun phrases
        for np in blob.noun_phrases:
            if len(np.split()) <= 3:
                aspects.append(np.strip())
        
        # Extract individual nouns
        for word, pos in blob.tags:
            if pos.startswith('NN') and len(word) > 2:
                aspects.append(word.lower())
        
        # Remove duplicates
        seen = set()
        unique_aspects = []
        for asp in aspects:
            if asp not in seen:
                seen.add(asp)
                unique_aspects.append(asp)
        
        return unique_aspects if unique_aspects else ['general']
    
    def train_aspect_extractor_transformer(self, train_df, model_name, output_dir):
        
        # RoBERTa tokenizer needs add_prefix_space=True for pretokenized inputs
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
                        label_ids.append(self.aspect_label_map[label[word_idx]])
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
            num_labels=len(self.aspect_label_map)
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
    
    def extract_aspects_transformer(self, text, model, tokenizer):
        
        tokens = text.split()
        inputs = tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=128
        )
        
        # Move inputs to the same device as the model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=2)
        
        word_ids = inputs.word_ids()
        predicted_labels = predictions[0].tolist()
        
        aspects = []
        current_aspect = []
        
        for word_idx, label_id in zip(word_ids, predicted_labels):
            if word_idx is None:
                continue
            
            label = self.reverse_aspect_label_map.get(label_id, 'O')
            
            if label == 'B-ASPECT':
                if current_aspect:
                    aspects.append(' '.join(current_aspect))
                current_aspect = [tokens[word_idx]]
            elif label == 'I-ASPECT' and current_aspect:
                current_aspect.append(tokens[word_idx])
            elif label == 'O' and current_aspect:
                aspects.append(' '.join(current_aspect))
                current_aspect = []
        
        if current_aspect:
            aspects.append(' '.join(current_aspect))
        
        return aspects if aspects else ['general']
    
    def calculate_aspect_extraction_metrics(self, extracted, true):
        total_extracted = 0
        total_true = 0
        total_correct = 0
        
        for ext_list, true_list in zip(extracted, true):
            ext_set = set([asp.lower() for asp in ext_list])
            true_set = set([asp.lower() for asp in true_list])
            
            total_extracted += len(ext_set)
            total_true += len(true_set)
            total_correct += len(ext_set & true_set)
        
        precision = total_correct / total_extracted if total_extracted > 0 else 0
        recall = total_correct / total_true if total_true > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return precision, recall, f1

    
    def train_sentiment_classifier_transformer(self, train_df, test_df, model_name, output_dir, method_name):
        
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
        
        train_data = train_data.map(lambda x: {'label': self.label_map[x['label']]})
        test_data = test_data.map(lambda x: {'label': self.label_map[x['label']]})
        
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
        self.loss_histories[method_name] = loss_callback.train_losses
        
        # Get predictions
        predictions_output = trainer.predict(test_dataset)
        predictions = np.argmax(predictions_output.predictions, axis=1)
        predictions = [self.reverse_label_map[pred] for pred in predictions]
        
        return model, tokenizer, predictions

    
    def method1_vader(self, train_df, test_df):
        start_time = time.time()
        analyzer = SentimentIntensityAnalyzer()
        
        extracted_aspects = []
        true_aspects = []
        
        for _, row in test_df.iterrows():
            sentence = str(row['english_translation'])
            aspects = self.extract_aspects_rule_based(sentence)
            extracted_aspects.append(aspects)
            
            if not pd.isna(row['aspects']):
                true_aspects.append([asp.strip() for asp in str(row['aspects']).split(',')])
            else:
                true_aspects.append([])
        
        self.aspect_extraction_results['VADER'] = {
            'extracted': extracted_aspects,
            'true': true_aspects
        }
        all_predictions = []
        all_true_labels = []
        
        for _, row in self.test_sentiment_df.iterrows():
            sentence = str(row['sentence_with_aspect'])
            scores = analyzer.polarity_scores(sentence)
      
            if scores['compound'] >= 0.05:
                pred = 'positive'
            elif scores['compound'] <= -0.05:
                pred = 'negative'
            else:
                pred = 'neutral'
            
            all_predictions.append(pred)
            all_true_labels.append(row['polarity'])
        
        training_time = time.time() - start_time
        print(f"Total time: {training_time:.4f} seconds")
        
        return all_predictions, all_true_labels, training_time
    
    def method2_textblob(self, train_df, test_df):
        start_time = time.time()
        extracted_aspects = []
        true_aspects = []
        
        for _, row in test_df.iterrows():
            sentence = str(row['english_translation'])
            aspects = self.extract_aspects_rule_based(sentence)
            extracted_aspects.append(aspects)
            
            if not pd.isna(row['aspects']):
                true_aspects.append([asp.strip() for asp in str(row['aspects']).split(',')])
            else:
                true_aspects.append([])
        
        self.aspect_extraction_results['TextBlob'] = {
            'extracted': extracted_aspects,
            'true': true_aspects
        }
        
        all_predictions = []
        all_true_labels = []
        
        for _, row in self.test_sentiment_df.iterrows():
            sentence = str(row['sentence_with_aspect'])
            blob = TextBlob(sentence)
            polarity = blob.sentiment.polarity
            
            if polarity > 0.1:
                pred = 'positive'
            elif polarity < -0.1:
                pred = 'negative'
            else:
                pred = 'neutral'
            
            all_predictions.append(pred)
            all_true_labels.append(row['polarity'])
        
        training_time = time.time() - start_time
        print(f"Total time: {training_time:.4f} seconds")
        
        return all_predictions, all_true_labels, training_time

    
    def method3_tfidf_lr(self, train_df, test_df):
        start_time = time.time()
        extracted_aspects = []
        true_aspects = []
        
        for _, row in test_df.iterrows():
            sentence = str(row['english_translation'])
            aspects = self.extract_aspects_rule_based(sentence)
            extracted_aspects.append(aspects)
            
            if not pd.isna(row['aspects']):
                true_aspects.append([asp.strip() for asp in str(row['aspects']).split(',')])
            else:
                true_aspects.append([])
        
        self.aspect_extraction_results['TF-IDF + LR'] = {
            'extracted': extracted_aspects,
            'true': true_aspects
        }
        
        X_train = self.train_sentiment_df['sentence_with_aspect'].values
        y_train = self.train_sentiment_df['polarity'].values
        X_test = self.test_sentiment_df['sentence_with_aspect'].values
        y_test = self.test_sentiment_df['polarity'].values
        
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train_vec, y_train)
        
        model_path = 'models/tfidf_lr_model.pkl'
        vectorizer_path = 'models/tfidf_vectorizer.pkl'
        joblib.dump(model, model_path)
        joblib.dump(vectorizer, vectorizer_path)
        
        predictions = model.predict(X_test_vec)
        
        training_time = time.time() - start_time
        print(f"Total time: {training_time:.4f} seconds")
        
        return predictions.tolist(), y_test.tolist(), training_time

    
    def method4_bert(self, train_df, test_df):
        try:
            start_time = time.time()
            model_name = "distilbert-base-uncased"
            
            aspect_model, aspect_tokenizer = self.train_aspect_extractor_transformer(
                self.train_extraction_df,
                model_name,
                './results_bert_aspect'
            )
            
            aspect_model.save_pretrained('models/bert_aspect_extractor')
            aspect_tokenizer.save_pretrained('models/bert_aspect_extractor')
            
            print("[Task 1: Extracting Aspects]")
            extracted_aspects = []
            true_aspects = []
            
            for _, row in test_df.iterrows():
                sentence = str(row['english_translation'])
                aspects = self.extract_aspects_transformer(sentence, aspect_model, aspect_tokenizer)
                extracted_aspects.append(aspects)
                
                if not pd.isna(row['aspects']):
                    true_aspects.append([asp.strip() for asp in str(row['aspects']).split(',')])
                else:
                    true_aspects.append([])
            
            self.aspect_extraction_results['BERT'] = {
                'extracted': extracted_aspects,
                'true': true_aspects
            }

            sentiment_model, sentiment_tokenizer, predictions = self.train_sentiment_classifier_transformer(
                self.train_sentiment_df,
                self.test_sentiment_df,
                model_name,
                './results_bert_sentiment',
                'BERT'
            )
            
            sentiment_model.save_pretrained('models/bert_sentiment')
            sentiment_tokenizer.save_pretrained('models/bert_sentiment')
            
            true_labels = self.test_sentiment_df['polarity'].tolist()
            
            training_time = time.time() - start_time
            print(f"Total time: {training_time:.4f} seconds")
            
            return predictions, true_labels, training_time
            
        except Exception as e:
            print(f"Error in BERT method: {e}")
            import traceback
            traceback.print_exc()
            return [], [], 0

    def method5_roberta(self, train_df, test_df):

        try:
            start_time = time.time()
            model_name = "roberta-base"
            
            aspect_model, aspect_tokenizer = self.train_aspect_extractor_transformer(
                self.train_extraction_df,
                model_name,
                './results_roberta_aspect'
            )
            
            aspect_model.save_pretrained('models/roberta_aspect_extractor')
            aspect_tokenizer.save_pretrained('models/roberta_aspect_extractor')

            extracted_aspects = []
            true_aspects = []
            
            for _, row in test_df.iterrows():
                sentence = str(row['english_translation'])
                aspects = self.extract_aspects_transformer(sentence, aspect_model, aspect_tokenizer)
                extracted_aspects.append(aspects)
                
                if not pd.isna(row['aspects']):
                    true_aspects.append([asp.strip() for asp in str(row['aspects']).split(',')])
                else:
                    true_aspects.append([])
            
            self.aspect_extraction_results['RoBERTa'] = {
                'extracted': extracted_aspects,
                'true': true_aspects
            }

            sentiment_model, sentiment_tokenizer, predictions = self.train_sentiment_classifier_transformer(
                self.train_sentiment_df,
                self.test_sentiment_df,
                model_name,
                './results_roberta_sentiment',
                'RoBERTa'
            )
            
            sentiment_model.save_pretrained('models/roberta_sentiment')
            sentiment_tokenizer.save_pretrained('models/roberta_sentiment')
            
            true_labels = self.test_sentiment_df['polarity'].tolist()
            
            training_time = time.time() - start_time
            print(f"Total time: {training_time:.4f} seconds")
            
            return predictions, true_labels, training_time
            
        except Exception as e:
            print(f"Error in RoBERTa method: {e}")
            import traceback
            traceback.print_exc()
            return [], [], 0
    
  
    def method6_electra(self, train_df, test_df):
        try:
            start_time = time.time()
            model_name = "google/electra-small-discriminator"

            aspect_model, aspect_tokenizer = self.train_aspect_extractor_transformer(
                self.train_extraction_df,
                model_name,
                './results_electra_aspect'
            )
            
            aspect_model.save_pretrained('models/electra_aspect_extractor')
            aspect_tokenizer.save_pretrained('models/electra_aspect_extractor')

            extracted_aspects = []
            true_aspects = []
            
            for _, row in test_df.iterrows():
                sentence = str(row['english_translation'])
                aspects = self.extract_aspects_transformer(sentence, aspect_model, aspect_tokenizer)
                extracted_aspects.append(aspects)
                
                if not pd.isna(row['aspects']):
                    true_aspects.append([asp.strip() for asp in str(row['aspects']).split(',')])
                else:
                    true_aspects.append([])
            
            self.aspect_extraction_results['ELECTRA'] = {
                'extracted': extracted_aspects,
                'true': true_aspects
            }
            

            sentiment_model, sentiment_tokenizer, predictions = self.train_sentiment_classifier_transformer(
                self.train_sentiment_df,
                self.test_sentiment_df,
                model_name,
                './results_electra_sentiment',
                'ELECTRA'
            )
            
            sentiment_model.save_pretrained('models/electra_sentiment')
            sentiment_tokenizer.save_pretrained('models/electra_sentiment')
            
            true_labels = self.test_sentiment_df['polarity'].tolist()
            
            training_time = time.time() - start_time
            print(f"Total time: {training_time:.4f} seconds")
            
            return predictions, true_labels, training_time
            
        except Exception as e:
            print(f"Error in ELECTRA method: {e}")
            import traceback
            traceback.print_exc()
            return [], [], 0

    
    def method7_deberta(self, train_df, test_df):
        try:
            start_time = time.time()
            model_name = "microsoft/deberta-v3-small"
            
            aspect_model, aspect_tokenizer = self.train_aspect_extractor_transformer(
                self.train_extraction_df,
                model_name,
                './results_deberta_aspect'
            )
            
            aspect_model.save_pretrained('models/deberta_aspect_extractor')
            aspect_tokenizer.save_pretrained('models/deberta_aspect_extractor')
  
            extracted_aspects = []
            true_aspects = []
            
            for _, row in test_df.iterrows():
                sentence = str(row['english_translation'])
                aspects = self.extract_aspects_transformer(sentence, aspect_model, aspect_tokenizer)
                extracted_aspects.append(aspects)
                
                if not pd.isna(row['aspects']):
                    true_aspects.append([asp.strip() for asp in str(row['aspects']).split(',')])
                else:
                    true_aspects.append([])
            
            self.aspect_extraction_results['DeBERTa'] = {
                'extracted': extracted_aspects,
                'true': true_aspects
            }
            
            sentiment_model, sentiment_tokenizer, predictions = self.train_sentiment_classifier_transformer(
                self.train_sentiment_df,
                self.test_sentiment_df,
                model_name,
                './results_deberta_sentiment',
                'DeBERTa'
            )
            
            sentiment_model.save_pretrained('models/deberta_sentiment')
            sentiment_tokenizer.save_pretrained('models/deberta_sentiment')
            
            true_labels = self.test_sentiment_df['polarity'].tolist()
            
            training_time = time.time() - start_time
            print(f"Total time: {training_time:.4f} seconds")
            
            return predictions, true_labels, training_time
            
        except Exception as e:
            print(f"Error in DeBERTa method: {e}")
            import traceback
            traceback.print_exc()
            return [], [], 0
    
    
    def method8_xlnet(self, train_df, test_df):
        """XLNet: NER aspect extraction + Sentiment classification"""
        try:
            start_time = time.time()
            model_name = "xlnet-base-cased"
            
            print("\n[Task 1: Training Aspect Extractor]")
            aspect_model, aspect_tokenizer = self.train_aspect_extractor_transformer(
                self.train_extraction_df,
                model_name,
                './results_xlnet_aspect'
            )
            
            aspect_model.save_pretrained('models/xlnet_aspect_extractor')
            aspect_tokenizer.save_pretrained('models/xlnet_aspect_extractor')
            
            print("[Task 1: Extracting Aspects]")
            extracted_aspects = []
            true_aspects = []
            
            for _, row in test_df.iterrows():
                sentence = str(row['english_translation'])
                aspects = self.extract_aspects_transformer(sentence, aspect_model, aspect_tokenizer)
                extracted_aspects.append(aspects)
                
                if not pd.isna(row['aspects']):
                    true_aspects.append([asp.strip() for asp in str(row['aspects']).split(',')])
                else:
                    true_aspects.append([])
            
            self.aspect_extraction_results['XLNet'] = {
                'extracted': extracted_aspects,
                'true': true_aspects
            }
            
            print("[Task 2: Training Sentiment Classifier]")
            sentiment_model, sentiment_tokenizer, predictions = self.train_sentiment_classifier_transformer(
                self.train_sentiment_df,
                self.test_sentiment_df,
                model_name,
                './results_xlnet_sentiment',
                'XLNet'
            )
            
            sentiment_model.save_pretrained('models/xlnet_sentiment')
            sentiment_tokenizer.save_pretrained('models/xlnet_sentiment')
            
            true_labels = self.test_sentiment_df['polarity'].tolist()
            
            training_time = time.time() - start_time
            print(f"Total time: {training_time:.4f} seconds")
            
            return predictions, true_labels, training_time
            
        except Exception as e:
            print(f"Error in XLNet method: {e}")
            import traceback
            traceback.print_exc()
            return [], [], 0
    
    def evaluate_method(self, y_true, y_pred, method_name, training_time):
        accuracy = accuracy_score(y_true, y_pred)
        labels = sorted(list(set(y_true) | set(y_pred)))
        
        precision = precision_score(y_true, y_pred, average='weighted', labels=labels, zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', labels=labels, zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', labels=labels, zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', labels=labels, zero_division=0)
        
        print(f"\n[SENTIMENT CLASSIFICATION METRICS]")
        print(f"Accuracy:            {accuracy:.4f}")
        print(f"Precision (Weighted): {precision:.4f}")
        print(f"Recall (Weighted):    {recall:.4f}")
        print(f"F1-Score (Weighted):  {f1:.4f}")
        print(f"F1-Score (Macro):     {f1_macro:.4f}")
        
        # Aspect extraction metrics
        if method_name in self.aspect_extraction_results:
            ext_data = self.aspect_extraction_results[method_name]
            asp_precision, asp_recall, asp_f1 = self.calculate_aspect_extraction_metrics(
                ext_data['extracted'], 
                ext_data['true']
            )
            
            print(f"\n[ASPECT EXTRACTION METRICS]")
            print(f"Precision: {asp_precision:.4f}")
            print(f"Recall:    {asp_recall:.4f}")
            print(f"F1-Score:  {asp_f1:.4f}")
        else:
            asp_precision = asp_recall = asp_f1 = 0.0
        
        print(f"\n[TRAINING TIME]")
        print(f"Total Time: {training_time:.4f} seconds")
        print('='*80)
        
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        return {
            'method': method_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'f1_macro': f1_macro,
            'aspect_precision': asp_precision,
            'aspect_recall': asp_recall,
            'aspect_f1': asp_f1,
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
        comparison_df = pd.DataFrame(self.results)
        pastel_colors = ['#FFB3BA', '#BAFFC9', '#BAE1FF', '#FFFFBA', '#E0BBE4', '#FFD9B3', '#C9FFE5', '#E6B3FF']
        
        # 1. SENTIMENT CLASSIFICATION METRICS (4 subplots)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Sentiment Classification Performance', fontsize=16, fontweight='bold')
        
        metrics = [
            ('accuracy', 'Accuracy', axes[0, 0]),
            ('precision', 'Precision', axes[0, 1]),
            ('recall', 'Recall', axes[1, 0]),
            ('f1', 'F1-Score', axes[1, 1])
        ]
        
        for metric, title, ax in metrics:
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
        plt.savefig('sentiment_classification_metrics.png', dpi=300, bbox_inches='tight')
        print("\nSaved: sentiment_classification_metrics.png")
        plt.show()
        
        # 2. ASPECT EXTRACTION METRICS (3 subplots)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Aspect Extraction Performance', fontsize=16, fontweight='bold')
        
        aspect_metrics = [
            ('aspect_precision', 'Precision', axes[0]),
            ('aspect_recall', 'Recall', axes[1]),
            ('aspect_f1', 'F1-Score', axes[2])
        ]
        
        for metric, title, ax in aspect_metrics:
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
        plt.savefig('aspect_extraction_metrics.png', dpi=300, bbox_inches='tight')
        print("Saved: aspect_extraction_metrics.png")
        plt.show()
        
        # 3. TRAINING TIME COMPARISON
        plt.figure(figsize=(12, 6))
        bars = plt.bar(comparison_df['method'], comparison_df['training_time'], 
                       color=pastel_colors[:len(comparison_df)])
        plt.ylabel('Training Time (seconds)', fontsize=12)
        plt.title('Training Time Comparison (Aspect Extraction + Sentiment Classification)', 
                  fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}s', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('training_time_comparison.png', dpi=300, bbox_inches='tight')
        print("Saved: training_time_comparison.png")
        plt.show()
        
        # 4. TRAINING LOSS CURVES (for transformer models)
        if self.loss_histories:
            plt.figure(figsize=(14, 8))
            
            for method_name, losses in self.loss_histories.items():
                if losses:
                    plt.plot(losses, label=method_name, linewidth=2, marker='o', markersize=4)
            
            plt.xlabel('Training Steps', fontsize=12)
            plt.ylabel('Loss', fontsize=12)
            plt.title('Sentiment Classifier Training Loss Curves', fontsize=14, fontweight='bold')
            plt.legend(fontsize=10, loc='best')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('training_loss_curves.png', dpi=300, bbox_inches='tight')
            print("Saved: training_loss_curves.png")
            plt.show()
        
        # 5. CONFUSION MATRICES
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
            plt.title(f'{result["method"]} - Confusion Matrix', 
                     fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            filename = f'confusion_matrix_{result["method"].replace(" ", "_").replace("+", "").lower()}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")
            plt.show()
        
        # 6. SUMMARY TABLES
        print("\n[SENTIMENT CLASSIFICATION METRICS]")
        sentiment_df = comparison_df[['method', 'accuracy', 'precision', 'recall', 'f1', 'f1_macro']].copy()
        print(sentiment_df.to_string(index=False))
        
        print("\n[ASPECT EXTRACTION METRICS]")
        aspect_df = comparison_df[['method', 'aspect_precision', 'aspect_recall', 'aspect_f1']].copy()
        print(aspect_df.to_string(index=False))
        
        print("\n[TRAINING TIME]")
        time_df = comparison_df[['method', 'training_time']].copy()
        time_df['training_time'] = time_df['training_time'].apply(lambda x: f'{x:.4f}s')
        print(time_df.to_string(index=False))
        
        print("="*140)
