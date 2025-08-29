from collections import Counter

import torch
from torch.utils.data import Dataset
from torchtext.vocab import vocab

from .constants import *


class NerDataset(Dataset):
    def __init__(self, filepath, vocab=None, label=None):
        self.data = None
        self.vocab = vocab
        self.label = label

        self.read_data(filepath)

    def read_data(self, filepath):
        all_words = []
        all_ners = []

        sents = open(filepath, 'r', encoding='utf-8').read().split('\n\n')
        for sent in sents:
            items = sent.split('\n')
            for item in items:
                if not item:
                    continue
                w, n = item.split('\t')
                if n[2:] in ['AGE', 'DATE', 'GENDER', 'JOB', 'LOCATION', 'NAME', 'ORGANIZATION', 'PATIENT_ID',
                             'SYMPTOM_AND_DISEASE', 'TRANSPORTATION']:
                    n = n[2:]
                w = '_'.join(w.split())
                all_words.append(w)
                all_ners.append(n)
        if self.vocab is None or self.label is None:
            self.vocab = vocab(Counter(all_words), specials=(PAD, UNK, BOS, EOS), special_first=False)
            self.label = vocab(Counter(all_ners), specials=(BOS, EOS, PAD), special_first=False)
            self.vocab.set_default_index(self.vocab[UNK])
            self.label.set_default_index(self.label['O'])

        X = []
        Y = []
        for sent in sents:
            x = [self.vocab[BOS]]
            y = [self.label[BOS]]
            items = sent.split('\n')
            for item in items:
                if not item:
                    continue
                w, n = item.split('\t')
                w = '_'.join(w.split())
                if n[2:] in ['AGE', 'DATE', 'GENDER', 'JOB', 'LOCATION', 'NAME', 'ORGANIZATION', 'PATIENT_ID',
                             'SYMPTOM_AND_DISEASE', 'TRANSPORTATION']:
                    n = n[2:]
                x.append(self.vocab[w])
                y.append(self.label[n])
            x.append(self.vocab[BOS])
            y.append(self.label[EOS])
            X.append(torch.tensor(x))
            Y.append(torch.tensor(y))

        self.data = []
        for x, y in zip(X, Y):
            self.data.append((x, y))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]