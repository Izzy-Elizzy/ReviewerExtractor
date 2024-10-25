import os
import json
from transformers import (
    AutoModelForCausalLM, 
    AutoModelForSeq2SeqLM, 
    AutoConfig,
    PreTrainedModel
)
from typing import Dict, Any, Union, Type
from pathlib import Path
from huggingface_hub import snapshot_download
import logging
import torch
import concurrent.futures
import pandas as pd

def detect_model_type(model_path: Union[str, Path]) -> Type[PreTrainedModel]:
    """
    Detect whether the model is a causal language model or sequence-to-sequence model.
    
    Args:
        model_path: Path to the local model directory
        
    Returns:
        Appropriate model class
    """
    try:
        config = AutoConfig.from_pretrained(model_path)
        
        architecture = config.architectures[0] if hasattr(config, 'architectures') else ""
        model_type = config.model_type
        
        seq2seq_architectures = [
            "T5", "BART", "MarianMT", "PegasusForConditionalGeneration",
            "MT5", "LED", "BLENDERBOT", "FSMTForConditionalGeneration"
        ]
        
        is_seq2seq = any(seq2seq_arch in architecture for seq2seq_arch in seq2seq_architectures) or \
                     any(seq2seq_arch.lower() in model_type.lower() for seq2seq_arch in seq2seq_architectures)
        
        return AutoModelForSeq2SeqLM if is_seq2seq else AutoModelForCausalLM
    except Exception as e:
        logging.error(f"Error detecting model type: {str(e)}")
        return AutoModelForCausalLM

def get_generation_parameters(model_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Detect and store generation parameters from a local model.
    
    Args:
        model_path: Path to the local model directory
        
    Returns:
        Dictionary containing generation-specific parameters
    """
    try:
        model_class = detect_model_type(model_path)
        config = AutoConfig.from_pretrained(model_path)
        model = model_class.from_pretrained(model_path)
        generation_config = model.generation_config if hasattr(model, 'generation_config') else None
        
        generation_params = {
            "max_length": (
                getattr(generation_config, 'max_length', None) or
                getattr(config, 'max_length', None) or
                1024
            ),
            "temperature": (
                getattr(generation_config, 'temperature', None) or
                getattr(config, 'temperature', None) or
                0.7
            ),
            "top_p": (
                getattr(generation_config, 'top_p', None) or
                getattr(config, 'top_p', None) or
                0.9
            ),
            "do_sample": (
                getattr(generation_config, 'do_sample', None) or
                getattr(config, 'do_sample', None) or
                True
            ),
            "repetition_penalty": (
                getattr(generation_config, 'repetition_penalty', None) or
                getattr(config, 'repetition_penalty', None) or
                1.2
            )
        }
        
        if model_class == AutoModelForSeq2SeqLM:
            seq2seq_params = {
                "min_length": (
                    getattr(generation_config, 'min_length', None) or
                    getattr(config, 'min_length', None) or
                    0
                ),
                "length_penalty": (
                    getattr(generation_config, 'length_penalty', None) or
                    getattr(config, 'length_penalty', None) or
                    1.0
                ),
                "num_beams": (
                    getattr(generation_config, 'num_beams', None) or
                    getattr(config, 'num_beams', None) or
                    4
                ),
                "early_stopping": (
                    getattr(generation_config, 'early_stopping', None) or
                    getattr(config, 'early_stopping', None) or
                    True
                )
            }
            generation_params.update(seq2seq_params)
        
        param_sources = {param: 'generation_config' if hasattr(generation_config, param)
                        else 'model_config' if hasattr(config, param)
                        else 'default'
                        for param in generation_params}
        
        return {
            "model_type": "Seq2SeqLM" if model_class == AutoModelForSeq2SeqLM else "CausalLM",
            "parameters": generation_params,
            "sources": param_sources
        }
        
    except Exception as e:
        logging.error(f"Error detecting generation parameters: {str(e)}")
        return {}

def save_model_config(model_path: Union[str, Path]) -> str:
    """
    Save model configuration to a JSON file in the model directory.
    
    Args:
        model_path: Path to the local model directory
        
    Returns:
        Path to the saved configuration file
    """
    model_path = Path(model_path)
    params = get_generation_parameters(model_path)
    
    if params:
        config_path = model_path / "generation_config.json"
        with open(config_path, 'w') as f:
            json.dump(params, f, indent=2)
        logging.info(f"Saved generation configuration to {config_path}")
        return str(config_path)
    return ""

def load_model_config(model_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load model configuration from the saved JSON file.
    
    Args:
        model_path: Path to the local model directory
        
    Returns:
        Dictionary containing model configuration
    """
    config_path = Path(model_path) / "generation_config.json"
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading model configuration: {str(e)}")
        return {}