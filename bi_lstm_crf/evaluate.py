#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BiLSTM-CRF Vietnamese NER Evaluation Script

This script evaluates a trained BiLSTM-CRF model on test data and computes
precision, recall, and F1-score for named entity recognition.
"""

import os
import sys
import argparse
import warnings
from pathlib import Path
from typing import List, Tuple, Dict

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
except ImportError as e:
    print(f"Error: Required PyTorch dependencies not found: {e}")
    print("Please install with: pip install torch")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    print("Warning: tqdm not found. Install with: pip install tqdm")
    # Create a simple replacement for tqdm
    def tqdm(iterable, total=None, desc=None):
        return iterable

# Import local modules
try:
    from bi_lstm_crf import BiLSTM_CRF_NER
    from constants import PAD, UNK, BOS, EOS
    from dataset import NerDataset
except ImportError as e:
    print(f"Error: Failed to import local modules: {e}")
    print("Please ensure you're running the script from the correct directory.")
    sys.exit(1)


def padding(sents, pad_idx, device):
    """
    Pad sentences to the same length for batch processing

    Args:
        sents: List of sentence tensors
        pad_idx: Padding token index
        device: Device to place tensors on

    Returns:
        Tuple of (padded_tensor, lengths)
    """
    if not sents:
        return torch.tensor([[]], dtype=torch.long, device=device), [0]

    lengths = [len(sent) for sent in sents]
    max_len = max(lengths)  # Fix: use max instead of lengths[0]

    padded_data = []
    for s in sents:
        sent_list = s.tolist() if hasattr(s, 'tolist') else s
        padded_sent = sent_list + [pad_idx] * (max_len - len(sent_list))
        padded_data.append(padded_sent)

    return torch.tensor(padded_data, dtype=torch.long, device=device), lengths


def _get_tags(sents):
    """
    Extract entity spans from IOB-tagged sentences

    Args:
        sents: List of sentences, each containing IOB tags

    Returns:
        Set of entity tuples (entity_type, start_idx, end_idx, sent_idx)
    """
    tags = []
    for sent_idx, iob_tags in enumerate(sents):
        curr_tag = {'type': None, 'start_idx': None,
                    'end_idx': None, 'sent_idx': None}

        for i, tag in enumerate(iob_tags):
            if tag == 'O':
                # End current entity if exists
                if curr_tag['type']:
                    tags.append(tuple(curr_tag.values()))
                    curr_tag = {'type': None, 'start_idx': None,
                                'end_idx': None, 'sent_idx': None}
            elif tag.startswith('B-'):
                # End previous entity if exists
                if curr_tag['type']:
                    tags.append(tuple(curr_tag.values()))

                # Start new entity
                curr_tag['type'] = tag[2:]
                curr_tag['start_idx'] = i
                curr_tag['end_idx'] = i
                curr_tag['sent_idx'] = sent_idx
            elif tag.startswith('I-'):
                # Continue current entity
                if curr_tag['type'] and curr_tag['type'] == tag[2:]:
                    curr_tag['end_idx'] = i
                else:
                    # Inconsistent tagging - treat as new entity
                    if curr_tag['type']:
                        tags.append(tuple(curr_tag.values()))
                    curr_tag['type'] = tag[2:]
                    curr_tag['start_idx'] = i
                    curr_tag['end_idx'] = i
                    curr_tag['sent_idx'] = sent_idx
            else:
                # Handle non-IOB format tags (direct entity labels)
                if curr_tag['type']:
                    tags.append(tuple(curr_tag.values()))

                if tag != 'O':  # Non-O tag without B- or I- prefix
                    curr_tag['type'] = tag
                    curr_tag['start_idx'] = i
                    curr_tag['end_idx'] = i
                    curr_tag['sent_idx'] = sent_idx
                else:
                    curr_tag = {'type': None, 'start_idx': None,
                                'end_idx': None, 'sent_idx': None}

        # Don't forget the last entity
        if curr_tag['type']:
            tags.append(tuple(curr_tag.values()))

    return set(tags)


def compute_metrics(y_true, y_pred):
    """
    Compute precision, recall, and F1-score for NER evaluation

    Args:
        y_true: List of true tag sequences
        y_pred: List of predicted tag sequences

    Returns:
        Dictionary with precision, recall, f1, and counts
    """
    tags_true = _get_tags(y_true)
    tags_pred = _get_tags(y_pred)

    ne_ref = len(tags_true)  # Number of true entities
    ne_sys = len(tags_pred)  # Number of predicted entities
    ne_true = len(tags_true.intersection(tags_pred))  # Number of correct predictions

    # Handle edge cases
    if ne_ref == 0 and ne_sys == 0:
        return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0,
                'true_entities': 0, 'pred_entities': 0, 'correct': 0}

    if ne_ref == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
                'true_entities': 0, 'pred_entities': ne_sys, 'correct': 0}

    if ne_sys == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
                'true_entities': ne_ref, 'pred_entities': 0, 'correct': 0}

    # Calculate metrics
    precision = ne_true / ne_sys if ne_sys > 0 else 0.0
    recall = ne_true / ne_ref if ne_ref > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_entities': ne_ref,
        'pred_entities': ne_sys,
        'correct': ne_true
    }


def f_measure(y_true, y_pred):
    """
    Backward compatibility function - returns only F1 score
    """
    metrics = compute_metrics(y_true, y_pred)
    return metrics['f1']


def collate_fn(samples):
    """
    Collate function for DataLoader - sorts samples by length for efficient padding
    """
    if not samples:
        return [], []

    samples = sorted(samples, key=lambda x: len(x[0]), reverse=True)
    sentences = [x[0] for x in samples]
    tags = [x[1] for x in samples]
    return sentences, tags


def evaluate_model(model_path: str, test_data_path: str, device: str = 'auto',
                  batch_size: int = 1, verbose: bool = True):
    """
    Evaluate a trained BiLSTM-CRF model on test data

    Args:
        model_path: Path to the trained model checkpoint
        test_data_path: Path to the test data file
        device: Device to use ('auto', 'cpu', 'cuda')
        batch_size: Batch size for evaluation
        verbose: Whether to print detailed results

    Returns:
        Dictionary with evaluation metrics
    """
    # Setup device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    if verbose:
        print(f"Using device: {device}")
        print(f"Loading model from: {model_path}")

    # Load model - use the model's built-in load method
    try:
        model = BiLSTM_CRF_NER.load(model_path)
        model.to(device)
        model.eval()

        # Extract vocabularies from the loaded model
        vocab = model.sent_vocab
        label = model.tag_vocab

        if verbose:
            print(f"Model loaded successfully!")
            print(f"Vocabulary size: {len(vocab)}")
            print(f"Label vocabulary size: {len(label)}")

    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

    # Load test dataset
    if verbose:
        print(f"Loading test data from: {test_data_path}")

    try:
        test_dataset = NerDataset(test_data_path, vocab, label)
        test_iter = DataLoader(test_dataset, batch_size, collate_fn=collate_fn)

        if verbose:
            print(f"Test dataset loaded: {len(test_dataset)} samples")

    except Exception as e:
        raise RuntimeError(f"Failed to load test data: {e}")

    # Evaluation loop
    y_test = []
    y_pred = []

    if verbose:
        print("Starting evaluation...")

    with torch.no_grad():
        for idx, (sentences, tags) in tqdm(enumerate(test_iter), total=len(test_iter),
                                          desc="Evaluating", disable=not verbose):
            try:
                # Pad sentences for batch processing
                sentences, sent_lengths = padding(sentences, label[PAD], device)

                # Get predictions from model
                pred_tags = model.predict(sentences, sent_lengths)

                # Process predictions and ground truth
                if pred_tags and len(pred_tags) > 0:
                    # Remove BOS and EOS tokens and convert to IOB format
                    pred_sequence = pred_tags[0]
                    if len(pred_sequence) >= 2:
                        pred_sequence = pred_sequence[1:-1]  # Remove BOS and EOS

                    pred_iob = model.iob_tag(pred_sequence)

                    # Process ground truth tags
                    true_sequence = tags[0].tolist()
                    if len(true_sequence) >= 2:
                        true_sequence = true_sequence[1:-1]  # Remove BOS and EOS

                    true_iob = model.iob_tag(true_sequence)

                    # Ensure same length
                    min_len = min(len(pred_iob), len(true_iob))
                    pred_iob = pred_iob[:min_len]
                    true_iob = true_iob[:min_len]

                    y_pred.append(pred_iob)
                    y_test.append(true_iob)

            except Exception as e:
                if verbose:
                    print(f"Error processing sample {idx}: {e}")
                continue

    # Compute metrics
    if not y_test or not y_pred:
        raise RuntimeError("No valid predictions were generated")

    metrics = compute_metrics(y_test, y_pred)

    if verbose:
        print("\nEvaluation Results:")
        print("-" * 40)
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1']:.4f}")
        print(f"\nEntity Counts:")
        print(f"True entities:      {metrics['true_entities']}")
        print(f"Predicted entities: {metrics['pred_entities']}")
        print(f"Correct entities:   {metrics['correct']}")

    return metrics


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(
        description="Evaluate BiLSTM-CRF Vietnamese NER Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate.py --test_data data/data/covid_test.txt
  python evaluate.py --model checkpoints/lstm_ner_5.pt --test_data data/data/covid_test.txt
  python evaluate.py --test_data data/data/covid_test.txt --device cuda --batch_size 8
        """
    )

    parser.add_argument('--model', '-m', type=str,
                       default='checkpoints/lstm_ner_10.pt',
                       help='Path to the trained model checkpoint (default: checkpoints/lstm_ner_10.pt)')
    parser.add_argument('--test_data', '-t', type=str, required=True,
                       help='Path to the test data file')
    parser.add_argument('--device', '-d', choices=['auto', 'cpu', 'cuda'],
                       default='auto', help='Device to use for evaluation (default: auto)')
    parser.add_argument('--batch_size', '-b', type=int, default=1,
                       help='Batch size for evaluation (default: 1)')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress verbose output')

    args = parser.parse_args()

    # Validate paths
    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)

    if not Path(args.test_data).exists():
        print(f"Error: Test data file not found: {args.test_data}")
        sys.exit(1)

    try:
        # Run evaluation
        metrics = evaluate_model(
            model_path=args.model,
            test_data_path=args.test_data,
            device=args.device,
            batch_size=args.batch_size,
            verbose=not args.quiet
        )

        # Return F1 score for backward compatibility
        if args.quiet:
            print(f"{metrics['f1']:.4f}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
