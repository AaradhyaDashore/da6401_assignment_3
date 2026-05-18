import spacy
import spacy.cli
from datasets import load_dataset
from collections import Counter
from typing import List, Tuple, Dict

class Multi30kDataset:
    def __init__(self, split: str = 'train') -> None:
        """
        Loads the Multi30k dataset and prepares tokenizers.
        """
        self.split = split
        
        # Fetching the dataset split safely
        hf_data = load_dataset("bentrevett/multi30k")
        self.dataset = hf_data[split]
        
        # Loading tokenizers
        try:
            self.spacy_de = spacy.load("de_core_news_sm")
        except OSError:
            spacy.cli.download("de_core_news_sm")
            self.spacy_de = spacy.load("de_core_news_sm")

        try:
            self.spacy_en = spacy.load("en_core_web_sm")
        except OSError:
            spacy.cli.download("en_core_web_sm")
            self.spacy_en = spacy.load("en_core_web_sm")

        # Standard tokens configuration
        self.pad_str = "<pad>"
        self.unk_str = "<unk>"
        self.sos_str = "<sos>"
        self.eos_str = "<eos>"
        
        self.special_tokens = [self.pad_str, self.unk_str, self.sos_str, self.eos_str]
        
        # Vocabulary dictionaries to be populated later
        self.src_vocab: Dict[str, int] = None
        self.tgt_vocab: Dict[str, int] = None

    def _tokenize_german(self, sentence: str) -> List[str]:
        """Helper to tokenize and lowercase German text."""
        return [token.text.lower() for token in self.spacy_de.tokenizer(sentence)]

    def _tokenize_english(self, sentence: str) -> List[str]:
        """Helper to tokenize and lowercase English text."""
        return [token.text.lower() for token in self.spacy_en.tokenizer(sentence)]
    
    def build_vocab(self) -> None:
        """
        Builds the vocabulary mapping for src (de) and tgt (en), including:
        <unk>, <pad>, <sos>, <eos>
        """
        german_counter = Counter()
        english_counter = Counter()

        # Count word frequencies across the entire dataset
        for row in self.dataset:
            de_toks = self._tokenize_german(row["de"])
            en_toks = self._tokenize_english(row["en"])
            
            german_counter.update(de_toks)
            english_counter.update(en_toks)

        # Initialize vocabularies with the special tokens
        self.src_vocab = {token: i for i, token in enumerate(self.special_tokens)}
        self.tgt_vocab = {token: i for i, token in enumerate(self.special_tokens)}

        # Populate remaining tokens for the source language (German)
        for word, count in german_counter.items():
            if word not in self.src_vocab:
                self.src_vocab[word] = len(self.src_vocab)

        # Populate remaining tokens for the target language (English)
        for word, count in english_counter.items():
            if word not in self.tgt_vocab:
                self.tgt_vocab[word] = len(self.tgt_vocab)

    def process_data(self) -> List[Tuple[List[int], List[int]]]:
        """
        Convert English and German sentences into integer token lists using
        spacy and the defined vocabulary. 
        """
        processed_pairs = []

        # Pre-fetch unknown token indices for speed
        unk_idx_src = self.src_vocab[self.unk_str]
        unk_idx_tgt = self.tgt_vocab[self.unk_str]

        for row in self.dataset:
            # Tokenize raw text
            de_toks = self._tokenize_german(row["de"])
            en_toks = self._tokenize_english(row["en"])

            # Map source words to indices
            german_indices = [
                self.src_vocab.get(word, unk_idx_src) for word in de_toks
            ]

            # Map target words to indices (Wrapped with SOS and EOS tokens)
            english_indices = [self.tgt_vocab[self.sos_str]]
            for word in en_toks:
                english_indices.append(self.tgt_vocab.get(word, unk_idx_tgt))
            english_indices.append(self.tgt_vocab[self.eos_str])

            processed_pairs.append((german_indices, english_indices))

        return processed_pairs