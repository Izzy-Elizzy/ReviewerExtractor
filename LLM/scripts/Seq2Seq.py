import torch
import pandas as pd
import os
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoConfig
)
import logging
import json

"""
Module: summarization_seq2seq.py

This module provides functionality for setting up a sequence-to-sequence (seq2seq) model for text summarization 
and generating summaries for a given set of input texts. It utilizes the Hugging Face Transformers library for 
model loading and text processing. The module also incorporates detailed logging for debugging and monitoring.
"""

def setup_logging():
    """
    Function: setup_logging

    Configures the logging system to output messages to both a log file ('model_processing.log') and the console.
    The log format includes timestamp, logger name, log level, and the message.

    Returns:
        logging.Logger: A configured logger object.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('model_processing.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def setup_model(model_path: str, config_path: str = None) -> tuple[AutoModelForSeq2SeqLM, AutoTokenizer, torch.device]:
    """
    Function: setup_model

    Loads a sequence-to-sequence (seq2seq) model and its tokenizer from Hugging Face's model hub.  Handles loading 
    configurations from either a separate config file or from the model directory itself.  Selects the appropriate 
    device (CUDA if available, otherwise CPU) for model execution.

    Args:
        model_path (str): Path to the directory containing the pre-trained model.
        config_path (str, optional): Path to a separate configuration file for the model. If None, loads the config 
                                      from the model directory. Defaults to None.

    Returns:
        tuple[AutoModelForSeq2SeqLM, AutoTokenizer, torch.device]: A tuple containing the loaded model, 
                                                                   tokenizer, and the device being used.

    Raises:
        Exception: Raises any exceptions encountered during model loading or device selection.  Exceptions are 
                   logged before being raised.

    """
    logger = logging.getLogger(__name__)
    try:
        # Load configuration
        if config_path:
            config = AutoConfig.from_pretrained(config_path)
            logger.info(f"Loaded model configuration from {config_path}")
        else:
            config = AutoConfig.from_pretrained(model_path)
            logger.warning("No configuration file provided. Loading from model path.")

        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.float16 # Use float16 for reduced memory usage
        )

        # Device selection
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        model.to(device)  # Move the model to the selected device

        return model, tokenizer, device

    except Exception as e:
        logger.error(f"Error setting up model: {str(e)}", exc_info=True)
        raise


def summarize_text(text: str, model: AutoModelForSeq2SeqLM, tokenizer: AutoTokenizer, device: torch.device, 
                   max_length: int = 3072, stride: int = 50) -> Union[str, None]:
    """
    Function: summarize_text

    Generates a summary for a given text using a seq2seq model.  Handles text longer than the model's maximum 
    input length by splitting it into overlapping chunks.

    Args:
        text (str): The input text to summarize.
        model (AutoModelForSeq2SeqLM): The pre-trained seq2seq model.
        tokenizer (AutoTokenizer): The tokenizer associated with the model.
        device (torch.device): The device (CPU or CUDA) on which to run the model.
        max_length (int, optional): The maximum sequence length for the model. Defaults to 3072.
        stride (int, optional): The stride used when splitting long texts into chunks. Defaults to 50.

    Returns:
        Union[str, None]: The generated summary as a string, or None if an error occurs.

    """
    try:
        # Encode text (handling long texts with truncation and stride)
        encoded_input = tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            stride=stride,
            return_overflowing_tokens=True,
            return_tensors='pt'
        )

        # Move tensors to device
        input_ids = encoded_input.input_ids.to(device)
        attention_mask = encoded_input.attention_mask.to(device)

        # Generate summary (using beam search for better quality)
        summary_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=max_length // 4, # Limit summary length to 1/4 of input
            min_length=50, # Minimum summary length
            no_repeat_ngram_size=3, # Prevent repetition of 3-grams
            num_beams=4, # Number of beams for beam search
            length_penalty=2.0, # Penalty for longer summaries
            early_stopping=True # Stop early if beam search converges
        )

        # Decode summary
        summary = tokenizer.batch_decode(
            summary_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        full_summary = " ".join(summary)
        return full_summary.strip()

    except Exception as e:
        logging.error(f"Error summarizing text: {str(e)}", exc_info=True)
        return None


def generate_summaries(model_path: str, config_path: str = None, papers: List[str] = None) -> Union[List[str], None]:
    """
    Function: generate_summaries

    Generates summaries for a list of input texts using a seq2seq model.  Handles potential errors during 
    both model setup and individual summarization. Cleans the CUDA cache after each paper to prevent memory issues 
    when using a GPU.

    Args:
        model_path (str): Path to the pre-trained seq2seq model.
        config_path (str, optional): Path to the model's configuration file. Defaults to None.
        papers (List[str], optional): A list of input texts to summarize. Defaults to None.

    Returns:
        Union[List[str], None]: A list of generated summaries, or None if an error occurs.
    """
    logger = setup_logging()

    try:
        model, tokenizer, device = setup_model(model_path, config_path)
        summaries = []
        for i, paper in enumerate(papers or []): #Added a check for papers being None
            try:
                logger.info(f"Processing paper {i + 1}/{len(papers)}")
                summary = summarize_text(paper, model, tokenizer, device)
                summaries.append(summary)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache() #Prevent GPU memory leaks
            except Exception as e:
                logger.error(f"Error processing paper {i + 1}: {str(e)}", exc_info=True)
                summaries.append(None)
        return summaries
    except Exception as e:
        logger.error(f"Error generating summaries: {str(e)}", exc_info=True)
        return None
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache() #Ensure GPU cache is cleared at the end


# Example usage (uncomment to run):
# if __name__ == "__main__":
#     model_path = "./models/led-large-book-summary"
#     config_path = "./models/led-large-book-summary/config.json"  # Optional config path

#     papers = [
#         "This is the text of the first paper...",
#         "This is the text of the second paper...",
#     ]

#     summaries = generate_summaries(model_path, config_path, papers)

#     if summaries:
#         for i, summary in enumerate(summaries):
#             if summary:
#                 print(f"Summary {i + 1}:\n{summary}\n")
#             else:
#                 print(f"Failed to generate summary {i + 1}")
#     else:
#         print("Failed to generate summaries.")