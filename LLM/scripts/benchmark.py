#!/usr/bin/env python3

import argparse
import sys
from typing import Dict, List
import numpy as np
from transformers import AutoTokenizer, AutoModel
from rouge_score import rouge_scorer
from bert_score import BERTScorer
import torch
import re
import json

"""
Module: scientific_metrics_evaluator.py

This module provides a class `ScientificMetricsEvaluator` for evaluating the quality of text summaries 
using a combination of ROUGE and BERTScore metrics, along with a novelty score based on n-gram analysis.  It's designed 
to assess the abstractiveness and quality of generated summaries compared to reference texts.
"""

class ScientificMetricsEvaluator:
    """
    Class: ScientificMetricsEvaluator

    This class implements methods for evaluating scientific text summaries using ROUGE, BERTScore, and n-gram novelty.

    Attributes:
        rouge_scorer (rouge_scorer.RougeScorer):  An instance of the ROUGE scorer for calculating ROUGE-1, ROUGE-2, 
                                                  ROUGE-L, and ROUGE-Lsum scores.  Stemming is enabled (`use_stemmer=True`) 
                                                  to improve matching across different word forms.
        bert_scorer (bert_score.BERTScorer): An instance of the BERTScorer for calculating precision, recall, and F1 scores 
                                               based on contextual embeddings.  It uses the `adsabs/astroBERT` model,
                                               which is specifically trained for scientific text.
        tokenizer (transformers.AutoTokenizer): A tokenizer for the `adsabs/astroBERT` model.  This is used to chunk 
                                                 long texts into smaller, manageable sequences for BERTScore calculation.
        max_length (int): The maximum length (in tokens) of a text sequence processed by the BERT scorer.  Longer texts 
                           are chunked into overlapping sequences.
        overlap (int): The amount of overlap (in tokens) between consecutive chunks of text. This helps to maintain context
                        across chunk boundaries.
    """
    def __init__(self):
        """
        Method: __init__

        Initializes the `ScientificMetricsEvaluator` by creating instances of the ROUGE scorer and BERTScorer, 
        and loading the tokenizer for the `adsabs/astroBERT` model.
        """
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL', 'rougeLsum'],
            use_stemmer=True
        )
        
        self.bert_scorer = BERTScorer(
            model_type="adsabs/astroBERT",
            num_layers=9,
            batch_size=32,
            nthreads=4,
            all_layers=False,
            idf=False,
            lang='en',
            rescale_with_baseline=False
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained("adsabs/astroBERT")
        self.max_length = 512
        self.overlap = 50

    def _preprocess_text_rouge(self, text: str) -> str:
        """
        Method: _preprocess_text_rouge

        Preprocesses the input text for ROUGE scoring.  Removes characters that are not alphanumeric, whitespace, 
        or common punctuation marks (.,!?-).  Preserves whitespace to maintain sentence structure for ROUGE.

        Args:
            text (str): The input text.

        Returns:
            str: The preprocessed text.
        """
        text = re.sub(r'[^\w\s,.!?-]', '', text)
        return ' '.join(text.split()).strip()

    def _preprocess_text_bert(self, text: str) -> str:
        """
        Method: _preprocess_text_bert

        Preprocesses the input text for BERTScore scoring.  Removes characters that are not alphanumeric or 
        common punctuation marks (.,!?-). This is a more minimal preprocessing step than for ROUGE.

        Args:
            text (str): The input text.

        Returns:
            str: The preprocessed text.
        """
        return re.sub(r'[^\w,.!?-]', '', text)

    def _chunk_text(self, text: str) -> List[str]:
        """
        Method: _chunk_text

        Chunks a long text into overlapping sequences of a maximum length suitable for the BERT scorer. This is 
        necessary because the BERTScorer has a limit on the length of input sequences.

        Args:
            text (str): The input text.

        Returns:
            List[str]: A list of text chunks.
        """
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        
        for i in range(0, len(tokens), self.max_length - self.overlap):
            chunk_tokens = tokens[i:i + self.max_length]
            chunk_tokens = [self.tokenizer.cls_token_id] + chunk_tokens + [self.tokenizer.sep_token_id]
            chunk = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk)
        
        return chunks

    def calculate_rouge(self, reference: str, candidate: str) -> Dict[str, float]:
        """
        Method: calculate_rouge

        Calculates ROUGE scores (rouge1, rouge2, rougeL, rougeLsum) between a reference text and a candidate text.

        Args:
            reference (str): The reference text.
            candidate (str): The candidate text (e.g., a generated summary).

        Returns:
            Dict[str, float]: A dictionary containing the ROUGE scores.
        """
        reference = self._preprocess_text_rouge(reference)
        candidate = self._preprocess_text_rouge(candidate)
        
        scores = self.rouge_scorer.score(reference, candidate)
        
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure,
            'rougeLsum': scores['rougeLsum'].fmeasure
        }

    def calculate_bertscore(self, reference: str, candidate: str) -> Dict[str, float]:
        """
        Method: calculate_bertscore

        Calculates BERTScore (precision, recall, F1) between a reference text and a candidate text.  Handles long texts
        by chunking them into smaller, overlapping sequences before scoring.

        Args:
            reference (str): The reference text.
            candidate (str): The candidate text.

        Returns:
            Dict[str, float]: A dictionary containing the BERTScore scores.
        """
        reference = self._preprocess_text_bert(reference)
        candidate = self._preprocess_text_bert(candidate)
        
        reference_chunks = self._chunk_text(reference)
        candidate_chunks = self._chunk_text(candidate)
        
        chunk_scores = []
        for ref_chunk in reference_chunks:
            for cand_chunk in candidate_chunks:
                P, R, F1 = self.bert_scorer.score([cand_chunk], [ref_chunk])
                chunk_scores.append((float(P[0]), float(R[0]), float(F1[0])))
        
        if chunk_scores:
            avg_precision = np.mean([score[0] for score in chunk_scores])
            avg_recall = np.mean([score[1] for score in chunk_scores])
            avg_f1 = np.mean([score[2] for score in chunk_scores])
        else:
            avg_precision = avg_recall = avg_f1 = 0.0
        
        return {
            'precision': avg_precision,
            'recall': avg_recall,
            'f1': avg_f1
        }

    def calculate_ngram_novelty(self, reference: str, candidate: str) -> float:
        """
        Method: calculate_ngram_novelty

        Calculates a novelty score based on the proportion of n-grams (unigrams, bigrams, trigrams) in the 
        candidate text that are not present in the reference text.  This measures how much new information 
        the candidate text provides compared to the reference.

        Args:
            reference (str): The reference text.
            candidate (str): The candidate text.

        Returns:
            float: The novelty score (average across unigrams, bigrams, and trigrams).
        """
        reference = self._preprocess_text_rouge(reference)
        candidate = self._preprocess_text_rouge(candidate)
        
        def get_ngrams(text, n):
            words = text.split()
            return set(' '.join(words[i:i+n]) for i in range(len(words)-n+1))
        
        novelty_scores = []
        for n in [1, 2, 3]:
            ref_ngrams = get_ngrams(reference, n)
            cand_ngrams = get_ngrams(candidate, n)
            novel_ngrams = cand_ngrams - ref_ngrams
            if cand_ngrams:
                novelty_scores.append(len(novel_ngrams) / len(cand_ngrams))
        
        return np.mean(novelty_scores) if novelty_scores else 0.0

    def evaluate_summary(self, reference: str, candidate: str) -> Dict[str, float]:
        """
        Method: evaluate_summary

        Calculates all the metrics (ROUGE, BERTScore, novelty) and combines them into a single "abstractive_score".

        Args:
            reference (str): The reference text.
            candidate (str): The candidate text.

        Returns:
            Dict[str, float]: A dictionary containing all the calculated scores, including the combined 
                              'abstractive_score'.
        """
        rouge_scores = self.calculate_rouge(reference, candidate)
        bert_scores = self.calculate_bertscore(reference, candidate)
        novelty_score = self.calculate_ngram_novelty(reference, candidate)
        
        final_scores = {
            'rouge1': rouge_scores['rouge1'],
            'rouge2': rouge_scores['rouge2'],
            'rougeL': rouge_scores['rougeL'],
            'rougeLsum': rouge_scores['rougeLsum'],
            'bertscore_precision': bert_scores['precision'],
            'bertscore_recall': bert_scores['recall'],
            'bertscore_f1': bert_scores['f1'],
            'ngram_novelty': novelty_score
        }
        
        weights = {
            'bertscore_f1': 0.5,
            'ngram_novelty': 0.5,
        }
        
        final_scores['abstractive_score'] = sum(
            final_scores[metric] * weight
            for metric, weight in weights.items()
        )
        
        return final_scores

def benchmark(paper: str, summary: str) -> float:
    """
    Function: benchmark

    Calculates and returns the 'abstractive_score' for a given paper and its summary.  This score combines 
    BERTScore F1 and n-gram novelty scores.

    Args:
        paper (str): The text of the original scientific paper.
        summary (str): The generated summary of the paper.

    Returns:
        float: The 'abstractive_score', a combined metric reflecting the quality of the summary.
    """
    evaluator = ScientificMetricsEvaluator()
    scores = evaluator.evaluate_summary(paper, summary)
    return scores['abstractive_score']