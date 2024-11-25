import os
import json
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoConfig,
    PreTrainedModel,
)
from typing import Dict, Any, Union, Type
from pathlib import Path
from huggingface_hub import snapshot_download
import logging
import torch
import concurrent.futures
import pandas as pd

# Global settings for generation
settings = {
    "token_batch_length": 3072,
    "batch_stride": 50,
    "max_len_ratio": 8,
    "parameters": {},  # Initialize an empty dictionary for parameters
    "arch": None,
}

# Path to local configs directory
CONFIGS_DIR = "./model_configs"
DEPRECATED_CONFIGS_DIR = "./model_configs/deprecated"

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
            "T5",
            "BART",
            "MarianMT",
            "PegasusForConditionalGeneration",
            "MT5",
            "LED",
            "BLENDERBOT",
            "FSMTForConditionalGeneration",
        ]

        is_seq2seq = any(
            seq2seq_arch in architecture for seq2seq_arch in seq2seq_architectures
        ) or any(
            seq2seq_arch.lower() in model_type.lower()
            for seq2seq_arch in seq2seq_architectures
        )

        return AutoModelForSeq2SeqLM if is_seq2seq else AutoModelForCausalLM
    except Exception as e:
        logging.error(f"Error detecting model type: {str(e)}")
        return AutoModelForCausalLM


def get_generation_parameters(model_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Detect and store relevant generation parameters from a local model.

    Args:
        model_path: Path to the local model directory

    Returns:
        Dictionary containing generation-specific parameters
    """
    try:
        model_class = detect_model_type(model_path)
        config = AutoConfig.from_pretrained(model_path)
        model = model_class.from_pretrained(model_path)
        generation_config = (
            model.generation_config if hasattr(model, "generation_config") else None
        )

        generation_params = {}
        if model_class == AutoModelForSeq2SeqLM:
            generation_params = {
                "min_length": getattr(generation_config, "min_length", 0),
                "max_length": getattr(generation_config, "max_length", 1024),
                "no_repeat_ngram_size": getattr(generation_config, "no_repeat_ngram_size", 3),
                "encoder_no_repeat_ngram_size": getattr(generation_config, "encoder_no_repeat_ngram_size", 3),
                "repetition_penalty": getattr(generation_config, "repetition_penalty", 1.2),
                "num_beams": getattr(generation_config, "num_beams", 4),
                "length_penalty": getattr(generation_config, "length_penalty", 1.0),
                "early_stopping": getattr(generation_config, "early_stopping", True),
                "do_sample": getattr(generation_config, "do_sample", True),
            }

        return generation_params

    except Exception as e:
        logging.error(f"Error detecting generation parameters: {str(e)}")
        return settings


def save_model_config(model_path: Union[str, Path], model_name: str) -> str:
    """
    Save model configuration to a JSON file in the configs directory.

    Args:
        model_path: Path to the local model directory
        model_name: Name of the model

    Returns:
        Path to the saved configuration file
    """
    config_path = os.path.join(CONFIGS_DIR, f"{model_name}_config.json")
    if os.path.exists(config_path):
        logging.warning(f"Configuration file already exists for {model_name} at {config_path}.")
        user_input = input(f"Overwrite existing configuration for {model_name}? (y/n): ")
        if user_input.lower() != 'y':
            return ""

        # Move existing config to deprecated folder with timestamp
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        deprecated_config_path = os.path.join(DEPRECATED_CONFIGS_DIR, f"{model_name}_config_{timestamp}.json")
        os.makedirs(DEPRECATED_CONFIGS_DIR, exist_ok=True)
        os.rename(config_path, deprecated_config_path)

    params = get_generation_parameters(model_path)

    if params:
        os.makedirs(CONFIGS_DIR, exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(params, f, indent=2)
        logging.info(f"Saved generation configuration to {config_path}")
        return config_path
    return ""


def load_model_config(config_path: str) -> Dict[str, Any]:
    """
    Load model configuration from the saved JSON file.

    Args:
        model_name: Name of the model

    Returns:
        Dictionary containing model configuration
    """
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading model configuration: {str(e)}")
        return {}


def get_model_config(model_path: Union[str, Path], model_name: str) -> Dict[str, Any]:
    """
    Gets the model configuration and merges it with global settings.

    Args:
        model_path: Path to the local model directory
        model_name: Name of the model

    Returns:
        Dictionary containing merged model configuration
    """
    config = load_model_config(model_name)
    if config:
        config = {
            "token_batch_length": settings["token_batch_length"],
            "batch_stride": settings["batch_stride"],
            "max_len_ratio": settings["max_len_ratio"],
            "parameters": config,
        }
        return config
    return {}

# # Example usage:
# model_path = "/path/to/your/model"  # Replace with your model path
# model_name = "my_model"  # Replace with your model name

# # Save the configuration
# config_path = save_model_config(model_path, model_name)

# # Load the configuration
# config = load_model_config(model_name)

# # Get the model configuration
# model_config = get_model_config(model_path, model_name)
# print(f"Model config: {model_config}")