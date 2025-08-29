#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import string
import sys
import argparse
from typing import List, Tuple
import re

from .crf import CRF_NER
from .feature_extractor import FeatureExtractor


class CRFInference:
    def __init__(self, model_path: str = None):
        """
        Initialize CRF inference with trained model
        
        Args:
            model_path: Path to the trained CRF model file
        """
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), 'model.crfsuite')
        
        self.model_path = model_path
        self.model = None
        self.feature_extractor = FeatureExtractor()
        self._load_model()
    
    def _load_model(self):
        """Load the trained CRF model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        try:
            self.model = CRF_NER.load(self.model_path)
            print(f"Model loaded successfully from: {self.model_path}")
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
    
    def predict_sentence(self, tokens: List[str]) -> List[Tuple[str, str, float]]:
        """
        Predict NER labels for a list of tokens.

        Args:
            tokens: List of word tokens.

        Returns:
            List of (word, predicted_label, confidence_score) tuples.
        """
        if not tokens:
            return []

        # Extract features for this sentence.
        sentence_data = [(token, 'O') for token in tokens]
        X_features = self.feature_extractor.sentence2features(sentence_data)

        # Set the features in the tagger to enable marginal probability calculation.
        self.model.tagger_.set(X_features)

        # Predict labels.
        predicted_labels = self.model.predict([X_features])[0]

        # Calculate confidence scores and combine results.
        result = []
        for i, token in enumerate(tokens):
            label = predicted_labels[i]
            confidence = self.model.tagger_.marginal(label, i)
            result.append((token, label, confidence))

        return result

    def predict_text(self, text: str) -> List[Tuple[str, str, float]]:
        """
        Predict NER labels for input text
        
        Args:
            text: Input Vietnamese text
            
        Returns:
            List of (word, predicted_label) tuples
        """
        if not text.strip():
            return []
        
        # Tokenize the input text
        tokens = self.tokenize_vietnamese(text)
        
        # Predict labels
        predictions = self.predict_sentence(tokens)
        
        return predictions
    
    def format_output(self, predictions: List[Tuple[str, str, float]], format_type: str = 'simple') -> str:
        """
        Format the prediction output.

        Args:
            predictions: List of (word, label, confidence) tuples.
            format_type: Output format ('simple', 'detailed', 'entities_only').

        Returns:
            Formatted output string.
        """
        if not predictions:
            return "No predictions available."

        if format_type == 'simple':
            lines = [f"{word}\t{label}" for word, label, _ in predictions]
            return '\n'.join(lines)

        elif format_type == 'detailed':
            output_lines = ["Word-Label Predictions:", "-" * 50]
            for word, label, confidence in predictions:
                if label != 'O':
                    output_lines.append(f"{word:<20} -> {label:<15} (ENTITY, conf: {confidence:.3f})")
                else:
                    output_lines.append(f"{word:<20} -> {label:<15} (conf: {confidence:.3f})")
            return '\n'.join(output_lines)

        elif format_type == 'entities_only':
            entities = [(word, label) for word, label, _ in predictions if label != 'O']
            if not entities:
                return "No named entities found."

            output_lines = ["Identified Named Entities:", "-" * 25]
            for word, label in entities:
                output_lines.append(f"{word} -> {label}")
            return '\n'.join(output_lines)
        
        else:
            return self.format_output(predictions, 'simple')


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(
        description="CRF Named Entity Recognition Inference for Vietnamese Text",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python infer.py --text "Tôi là sinh viên trường Đại học Bách khoa Hà Nội."
  python infer.py --interactive
  python infer.py --text "COVID-19 xuất hiện tại Việt Nam." --format detailed
        """
    )
    
    parser.add_argument('--text', '-t', type=str, 
                       help='Input Vietnamese text for NER prediction')
    parser.add_argument('--model', '-m', type=str, 
                       help='Path to the trained CRF model file')
    parser.add_argument('--format', '-f', choices=['simple', 'detailed', 'entities_only'],
                       default='simple', help='Output format')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Run in interactive mode')
    
    args = parser.parse_args()
    
    try:
        # Initialize inference
        inferencer = CRFInference(args.model)
        
        if args.interactive:
            # Interactive mode
            print("CRF NER Interactive Mode")
            print("Enter Vietnamese text for named entity recognition.")
            print("Type 'quit' or 'exit' to stop.\n")
            
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
                    predictions = inferencer.predict_text(text)
                    
                    # Display results
                    print("\nPredictions:")
                    print(inferencer.format_output(predictions, args.format))
                    print()
                    
                except KeyboardInterrupt:
                    print("\nGoodbye!")
                    break
                except Exception as e:
                    print(f"Error processing text: {e}\n")
        
        elif args.text:
            # Single text prediction
            predictions = inferencer.predict_text(args.text)
            print(inferencer.format_output(predictions, args.format))
        
        else:
            # No input provided
            print("Please provide input text with --text or use --interactive mode.")
            print("Use --help for more information.")
            sys.exit(1)
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
