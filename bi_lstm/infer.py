#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Vietnamese Named Entity Recognition (NER) Inference Script
BiLSTM Model Implementation

This script provides inference capabilities for a pre-trained BiLSTM model
for Vietnamese Named Entity Recognition. It supports multiple input modes and
output formats for production use.

Author: Generated for Vietnamese NER Project
Date: 2025-08-28
"""

import os
import sys
import argparse
import json
import string
import re
import time
import warnings
from typing import List, Tuple, Dict, Optional, Union
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torchtext.vocab import vocab
    from collections import Counter
except ImportError as e:
    print(f"Error: Required dependency not found: {e}")
    print("Please install required packages with: pip install torch torchtext")
    sys.exit(1)

# Import local modules
try:
    from .bi_lstm import BiLSTM_NER
    from .constants import PAD, UNK, BOS, EOS
except ImportError as e:
    print(f"Error: Failed to import local modules: {e}")
    print("Please ensure you're running the script from the correct directory.")
    sys.exit(1)


class BiLSTMInference:
    """
    Vietnamese NER inference class using BiLSTM model
    
    This class handles model loading, text preprocessing, inference,
    and output formatting for Vietnamese Named Entity Recognition.
    """
    
    def __init__(self, model_path: str, device: str = 'auto'):
        """
        Initialize BiLSTM inference with trained model
        
        Args:
            model_path: Path to the trained model checkpoint (.pt file)
            device: Device to run inference on ('cpu', 'cuda', or 'auto')
        """
        self.model_path = Path(model_path)
        self.device = self._setup_device(device)
        self.model = None
        self.sent_vocab = None
        self.tag_vocab = None
        
        self._load_model()
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device"""
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        torch_device = torch.device(device)
        print(f"Using device: {torch_device}")
        return torch_device
    
    def _load_model(self):
        """Load the trained BiLSTM model"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")
        
        try:
            print(f"Loading model from: {self.model_path}")
            
            # Load model using the class method
            self.model = BiLSTM_NER.load(str(self.model_path))
            self.model.to(self.device)
            self.model.eval()
            
            # Extract vocabularies
            self.sent_vocab = self.model.sent_vocab
            self.tag_vocab = self.model.tag_vocab
            
            print(f"Model loaded successfully!")
            print(f"Vocabulary size: {len(self.sent_vocab)}")
            print(f"Tag vocabulary size: {len(self.tag_vocab)}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def tokenize_vietnamese(self, text: str) -> List[str]:
        """
        Tokenize Vietnamese text for NER processing.
        This version handles complex patterns like date ranges and attached punctuation.

        Args:
            text: Input Vietnamese text

        Returns:
            List of tokens
        """
        if not text.strip():
            return []

        # Replace underscores, then use regex for tokenization
        text = text.strip().replace('_', ' ')

        # Regex to find words (sequences of word characters) or any single non-whitespace, non-word character.
        # This effectively separates words, numbers, and punctuation.
        # For example, "24-7," becomes ["24", "-", "7", ","]
        tokens = re.findall(r'\d+/\d+|(?:[A-Z]\.)+[A-Z]?|\w+|[^\w\s]', text)

        return [token for token in tokens if token.strip()]
    
    def preprocess_tokens(self, tokens: List[str]) -> Tuple[torch.Tensor, List[int]]:
        """
        Convert tokens to tensor format expected by the model
        
        Args:
            tokens: List of string tokens
            
        Returns:
            Tuple of (token_tensor, sentence_lengths)
        """
        if not tokens:
            return torch.tensor([[]], dtype=torch.long), [0]
        
        token_indices = [self.sent_vocab[BOS]]
        for token in tokens:
            if token in self.sent_vocab:
                token_indices.append(self.sent_vocab[token])
            else:
                token_indices.append(self.sent_vocab[UNK])
        token_indices.append(self.sent_vocab[EOS])
        
        token_tensor = torch.tensor([token_indices], dtype=torch.long)
        sentence_lengths = [len(token_indices)]
        
        return token_tensor, sentence_lengths

    def predict_sentence(self, tokens: List[str], confidence_threshold: float = 0.0) -> List[Tuple[str, str, float]]:
        """
        Predict NER labels for a list of tokens

        Args:
            tokens: List of word tokens
            confidence_threshold: Minimum confidence threshold for predictions

        Returns:
            List of (word, predicted_label, confidence) tuples
        """
        if not tokens:
            return []

        try:
            # Preprocess tokens
            token_tensor, sent_lengths = self.preprocess_tokens(tokens)
            token_tensor = token_tensor.to(self.device)

            with torch.no_grad():
                # Get logits from model
                logits = self.model(token_tensor, sent_lengths)

                # Apply softmax to get probabilities
                probabilities = F.softmax(logits, dim=-1)

                # Get predicted labels and confidence scores
                max_probs, predicted_indices = torch.max(probabilities, dim=-1)

                # Extract predictions for the first (and only) sentence
                if logits.size(0) > 0:
                    sentence_preds = predicted_indices[0]  # Shape: [seq_len]
                    sentence_probs = max_probs[0]  # Shape: [seq_len]

                    # Skip BOS and EOS tokens
                    if len(sentence_preds) >= 2:
                        sentence_preds = sentence_preds[1:-1]  # Remove BOS and EOS
                        sentence_probs = sentence_probs[1:-1]  # Remove BOS and EOS

                    # Convert indices to IOB format
                    predicted_labels = self.model.iob_tag(sentence_preds.cpu().tolist())
                    confidences = sentence_probs.cpu().tolist()

                    # Ensure we have the same number of tokens and labels
                    min_len = min(len(tokens), len(predicted_labels), len(confidences))

                    results = []
                    for i in range(min_len):
                        confidence = confidences[i]
                        if confidence >= confidence_threshold:
                            results.append((tokens[i], predicted_labels[i], confidence))
                        else:
                            results.append((tokens[i], 'O', confidence))

                    return results
                else:
                    # Return all tokens with 'O' labels if no predictions
                    return [(token, 'O', 0.0) for token in tokens]

        except Exception as e:
            print(f"Error during prediction: {e}")
            # Return all tokens with 'O' labels in case of error
            return [(token, 'O', 0.0) for token in tokens]

    def predict_text(self, text: str, confidence_threshold: float = 0.0) -> List[Tuple[str, str, float]]:
        """
        Predict NER labels for input text

        Args:
            text: Input Vietnamese text
            confidence_threshold: Minimum confidence threshold for predictions

        Returns:
            List of (word, predicted_label, confidence) tuples
        """
        if not text.strip():
            return []

        # Tokenize the input text
        tokens = self.tokenize_vietnamese(text)

        if not tokens:
            return []

        # Predict labels
        predictions = self.predict_sentence(tokens, confidence_threshold)

        return predictions

    def extract_entities(self, predictions: List[Tuple[str, str, float]]) -> List[Dict[str, Union[str, int, float]]]:
        """
        Extract named entities from predictions

        Args:
            predictions: List of (word, label, confidence) tuples

        Returns:
            List of entity dictionaries with text, label, start, end, confidence
        """
        entities = []
        current_entity = None

        for i, (word, label, confidence) in enumerate(predictions):
            if label.startswith('B-'):
                # Start of new entity
                if current_entity:
                    entities.append(current_entity)

                entity_type = label[2:]  # Remove 'B-' prefix
                current_entity = {
                    'text': word,
                    'label': entity_type,
                    'start': i,
                    'end': i + 1,
                    'confidence': confidence
                }

            elif label.startswith('I-') and current_entity:
                # Continuation of current entity
                entity_type = label[2:]  # Remove 'I-' prefix
                if entity_type == current_entity['label']:
                    current_entity['text'] += ' ' + word
                    current_entity['end'] = i + 1
                    current_entity['confidence'] = min(current_entity['confidence'], confidence)
                else:
                    # Different entity type, start new entity
                    entities.append(current_entity)
                    current_entity = {
                        'text': word,
                        'label': entity_type,
                        'start': i,
                        'end': i + 1,
                        'confidence': confidence
                    }

            elif label == 'O':
                # Outside any entity
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None

            else:
                # Handle other cases (non-IOB format)
                if label != 'O':
                    if current_entity:
                        entities.append(current_entity)

                    current_entity = {
                        'text': word,
                        'label': label,
                        'start': i,
                        'end': i + 1,
                        'confidence': confidence
                    }
                else:
                    if current_entity:
                        entities.append(current_entity)
                        current_entity = None

        # Don't forget the last entity
        if current_entity:
            entities.append(current_entity)

        return entities

    def format_output(self, predictions: List[Tuple[str, str, float]], format_type: str = 'simple') -> str:
        """
        Format the prediction output

        Args:
            predictions: List of (word, label, confidence) tuples
            format_type: Output format ('simple', 'detailed', 'entities_only', 'json')

        Returns:
            Formatted output string
        """
        if not predictions:
            return "No predictions available."

        if format_type == 'simple':
            # Simple word\tlabel format
            lines = [f"{word}\t{label}" for word, label, _ in predictions]
            return '\n'.join(lines)

        elif format_type == 'detailed':
            # Detailed format with entity highlighting and confidence
            output_lines = []
            output_lines.append("Word-Label Predictions:")
            output_lines.append("-" * 50)

            for word, label, confidence in predictions:
                if label != 'O':
                    output_lines.append(f"{word:<20} -> {label:<15} (ENTITY, conf: {confidence:.3f})")
                else:
                    output_lines.append(f"{word:<20} -> {label:<15} (conf: {confidence:.3f})")

            return '\n'.join(output_lines)

        elif format_type == 'entities_only':
            # Show only identified entities
            entities = self.extract_entities(predictions)
            if not entities:
                return "No named entities found."

            output_lines = ["Identified Named Entities:"]
            output_lines.append("-" * 30)

            for entity in entities:
                output_lines.append(f"'{entity['text']}' -> {entity['label']} (conf: {entity['confidence']:.3f})")

            return '\n'.join(output_lines)

        elif format_type == 'json':
            # JSON format output
            entities = self.extract_entities(predictions)
            result = {
                'text': ' '.join([word for word, _, _ in predictions]),
                'entities': entities,
                'word_predictions': [
                    {'word': word, 'label': label, 'confidence': confidence}
                    for word, label, confidence in predictions
                ]
            }
            return json.dumps(result, ensure_ascii=False, indent=2)

        else:
            return self.format_output(predictions, 'simple')

    def predict_from_file(self, file_path: str, confidence_threshold: float = 0.0) -> List[Tuple[str, str, float]]:
        """
        Predict NER labels for text from a file

        Args:
            file_path: Path to input text file
            confidence_threshold: Minimum confidence threshold for predictions

        Returns:
            List of (word, predicted_label, confidence) tuples
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()

            if not text:
                print(f"Warning: File {file_path} is empty or contains only whitespace.")
                return []

            return self.predict_text(text, confidence_threshold)

        except FileNotFoundError:
            raise FileNotFoundError(f"Input file not found: {file_path}")
        except Exception as e:
            raise RuntimeError(f"Error reading file {file_path}: {e}")


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(
        description="BiLSTM Named Entity Recognition Inference for Vietnamese Text",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python infer.py --text "Tôi là sinh viên trường Đại học Bách khoa Hà Nội."
  python infer.py --interactive
  python infer.py --text "COVID-19 xuất hiện tại Việt Nam." --format detailed
  python infer.py --file input.txt --format json --output results.json
  python infer.py --text "Bệnh nhân Nguyễn Văn A, 35 tuổi." --confidence 0.8
        """
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument('--text', '-t', type=str,
                           help='Input Vietnamese text for NER prediction')
    input_group.add_argument('--file', '-f', type=str,
                           help='Path to input text file')
    input_group.add_argument('--interactive', '-i', action='store_true',
                           help='Run in interactive mode')

    # Model and device options
    parser.add_argument('--model', '-m', type=str,
                       default='checkpoints/ulstm_ner_20.pt',
                       help='Path to the trained BiLSTM model checkpoint (default: checkpoints/ulstm_ner_20.pt)')
    parser.add_argument('--device', '-d', choices=['auto', 'cpu', 'cuda'],
                       default='auto', help='Device to run inference on (default: auto)')

    # Output options
    parser.add_argument('--format', choices=['simple', 'detailed', 'entities_only', 'json'],
                       default='simple', help='Output format (default: simple)')
    parser.add_argument('--output', '-o', type=str,
                       help='Output file path (if not specified, prints to console)')
    parser.add_argument('--confidence', '-c', type=float, default=0.0,
                       help='Confidence threshold for predictions (default: 0.0)')

    # Parse arguments
    args = parser.parse_args()

    # Validate arguments
    if not any([args.text, args.file, args.interactive]):
        print("Error: Please provide input text with --text, --file, or use --interactive mode.")
        print("Use --help for more information.")
        sys.exit(1)

    try:
        # Initialize inference
        print("Initializing BiLSTM NER model...")
        inferencer = BiLSTMInference(args.model, args.device)

        if args.interactive:
            # Interactive mode
            print("\n" + "="*60)
            print("BiLSTM NER Interactive Mode")
            print("Enter Vietnamese text for named entity recognition.")
            print("Type 'quit', 'exit', or 'q' to stop.")
            print("="*60 + "\n")

            while True:
                try:
                    text = input("Enter text: ").strip()
                    if text.lower() in ['quit', 'exit', 'q']:
                        print("Goodbye!")
                        break

                    if not text:
                        print("Please enter some text.\n")
                        continue

                    # Make prediction
                    predictions = inferencer.predict_text(text, args.confidence)

                    # Display results
                    print("\nPredictions:")
                    print("-" * 40)
                    output = inferencer.format_output(predictions, args.format)
                    print(output)
                    print()

                except KeyboardInterrupt:
                    print("\nGoodbye!")
                    break
                except Exception as e:
                    print(f"Error processing text: {e}\n")

        else:
            # Single prediction mode
            if args.text:
                predictions = inferencer.predict_text(args.text, args.confidence)
            elif args.file:
                predictions = inferencer.predict_from_file(args.file, args.confidence)

            # Format output
            output = inferencer.format_output(predictions, args.format)

            # Save or print output
            if args.output:
                try:
                    with open(args.output, 'w', encoding='utf-8') as f:
                        f.write(output)
                    print(f"Results saved to: {args.output}")
                except Exception as e:
                    print(f"Error saving output: {e}")
                    print("Printing to console instead:")
                    print(output)
            else:
                print(output)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
