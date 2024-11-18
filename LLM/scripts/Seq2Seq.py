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

def setup_logging():
    """Configure logging with detailed formatting."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('model_processing.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def setup_model(model_path, config_path=None):
    """Sets up the Seq2Seq model and tokenizer."""
    logger = logging.getLogger(__name__)
    try:
        # Load configuration
        if config_path:
            config = AutoConfig.from_pretrained(config_path)
            logger.info(f"Loaded model configuration from {config_path}")
        else:
            config = AutoConfig.from_pretrained(model_path)
            logger.warning("No configuration file provided. Loading from model path.")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load model
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.float16
        )

        # Device mapping
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            logger.info(f"Moving model to {device}")
            model = model.to(device)
        else:
            device = torch.device("cpu")
            logger.info("Using CPU")

        return model, tokenizer, device

    except Exception as e:
        logger.error(f"Error in setup_model: {str(e)}", exc_info=True)
        raise

def summarize_text(text, model, tokenizer, device, max_length=3072, stride=50):
    """Generate summary for a single text using Seq2Seq model."""
    try:
        # Encode text with truncation and overlapping windows
        encoded_input = tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            stride=stride,
            return_overflowing_tokens=True,
            return_tensors='pt'
        )

        # Move inputs to device
        input_ids = encoded_input.input_ids.to(device)
        attention_mask = encoded_input.attention_mask.to(device)

        # Generate summary
        summary_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=max_length // 4,  # Limit summary length
            min_length=50,
            no_repeat_ngram_size=3,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True
        )

        # Decode summary
        summary = tokenizer.batch_decode(
            summary_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        # Combine all chunks
        full_summary = " ".join(summary)
        return full_summary.strip()

    except Exception as e:
        logging.error(f"Error in summarize_text: {str(e)}", exc_info=True)
        return None

def generate_summaries(model_path, config_path, papers):
    """Generates summaries for an array of papers using a Seq2Seq model."""
    logger = setup_logging()

    try:
        # Setup model
        model, tokenizer, device = setup_model(model_path, config_path)
        
        summaries = []
        for i, paper in enumerate(papers):
            try:
                logger.info(f"Processing paper {i+1}/{len(papers)}")
                summary = summarize_text(paper, model, tokenizer, device)
                summaries.append(summary)
                
                # Clear CUDA cache if using GPU
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                logger.error(f"Error processing paper {i+1}: {str(e)}", exc_info=True)
                summaries.append(None)

        return summaries

    except Exception as e:
        logger.error(f"Error in generate_summaries: {str(e)}", exc_info=True)
        return None
    finally:
        # Clear CUDA cache after processing all papers
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    # Example usage
    model_path = "./models/led-large-book-summary"
    config_path = "./models/led-large-book-summary/config.json"
    
    papers = [
        "First paper text goes here...",
        "Second paper text goes here...",
    ]

    summaries = generate_summaries(model_path, config_path, papers)

    if summaries:
        for i, summary in enumerate(summaries):
            if summary:
                print(f"Summary {i + 1}:\n{summary}\n")
            else:
                print(f"Failed to generate summary {i + 1}")
    else:
        print("Failed to generate summaries.")