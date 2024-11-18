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

class ScientificMetricsEvaluator:
    def __init__(self):
        """Initialize scientific metrics"""
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
        """Preprocessing for ROUGE - keeps whitespace"""
        text = re.sub(r'[^\w\s,.!?-]', '', text)
        return ' '.join(text.split()).strip()

    def _preprocess_text_bert(self, text: str) -> str:
        """Minimal preprocessing for BERT"""
        return re.sub(r'[^\w,.!?-]', '', text)

    def _chunk_text(self, text: str) -> List[str]:
        """Chunk text into overlapping sequences"""
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        
        for i in range(0, len(tokens), self.max_length - self.overlap):
            chunk_tokens = tokens[i:i + self.max_length]
            chunk_tokens = [self.tokenizer.cls_token_id] + chunk_tokens + [self.tokenizer.sep_token_id]
            chunk = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk)
        
        return chunks

    def calculate_rouge(self, reference: str, candidate: str) -> Dict[str, float]:
        """Calculate ROUGE scores"""
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
        """Calculate BERTScore using chunked texts"""
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
        """Calculate the proportion of novel n-grams"""
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
        """Calculate all metrics"""
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

def benchmark(paper, summary):
        evaluator = ScientificMetricsEvaluator()
        scores = evaluator.evaluate_summary(paper, summary)

        return scores['abstractive_score']