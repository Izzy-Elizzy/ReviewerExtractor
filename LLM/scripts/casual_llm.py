import torch
import pandas as pd
import os
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    AutoConfig
)
from langchain import LLMChain, HuggingFacePipeline, PromptTemplate
import logging
import json
import TextAnalysis as TA  # Assuming this module contains n-gram analysis functions

"""
Module: summarization.py

This module provides functionality for setting up a causal language model (LLM) for text summarization and 
generating summaries for a given set of input texts.  It uses the Hugging Face Transformers library for model loading,
Langchain for prompt engineering and LLM interaction, and a custom TextAnalysis module (not shown here) for 
post-processing of the summaries.  The module also includes detailed logging for debugging and monitoring.
"""

def setup_logging():
    """
    Function: setup_logging

    Configures logging to write messages to both a log file ('model_processing.log') and the console.  
    The log format includes timestamp, logger name, log level, and the message itself.

    Returns:
        logging.Logger: The configured logger instance.
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

def setup_model(model_path: str, config_path: str = None) -> LLMChain:
    """
    Function: setup_model

    Loads a causal language model (LLM) from a given path, sets up a text generation pipeline using Hugging Face's 
    `pipeline` function, and creates a Langchain `LLMChain` for streamlined interaction with the model using a 
    specified prompt template.  Handles both model-specific and generic configurations.

    Args:
        model_path (str): The path to the pre-trained LLM model.
        config_path (str, optional): The path to a model configuration file. If None, the configuration is 
                                      loaded from the model path. Defaults to None.

    Returns:
        LLMChain: A Langchain LLMChain object ready for text summarization.  This object encapsulates the model, 
                   tokenizer, and the prompt template.

    Raises:
        Exception: If any error occurs during model loading or pipeline setup.  The exception is logged and 
                   re-raised.
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
        tokenizer.pad_token = tokenizer.eos_token  # Ensure pad token is set
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.float16, # Use float16 for reduced memory usage
            trust_remote_code=True # Allow loading models with custom code
        )

        # Device selection (GPU if available, otherwise CPU)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        model.to(device)


        # Create text generation pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            max_new_tokens=8192, # Maximum number of tokens to generate
            do_sample=True, # Use sampling for generation
            top_k=10, # Consider top 10 tokens during sampling
            num_return_sequences=1, # Return only one summary
            eos_token_id=tokenizer.eos_token_id, # Use end-of-sequence token
            device=device
        )

        # Integrate pipeline with Langchain for prompt management
        llm = HuggingFacePipeline(pipeline=pipe, model_kwargs={'temperature': 0.5}) # Temperature controls randomness

        # Define the prompt template
        template = """
        Write a summary of the following text delimited by triple backticks.
        Return your response which covers the key points of the text.
        ```{text}```
        SUMMARY:
        """
        prompt = PromptTemplate(template=template, input_variables=["text"])
        llm_chain = LLMChain(prompt=prompt, llm=llm)

        return llm_chain

    except Exception as e:
        logger.error(f"Error in setup_model: {str(e)}", exc_info=True)
        raise


def extract_summary(llm_chain_output: Dict[str, str]) -> str:
    """
    Function: extract_summary

    Extracts the generated summary from the output of the LLMChain.  The summary is expected to be 
    delimited by "SUMMARY:" in the output.

    Args:
        llm_chain_output (Dict[str, str]): The output dictionary from the LLMChain's `run` method.

    Returns:
        str: The extracted summary text.
    """
    full_output = llm_chain_output.get('text', '')
    summary_parts = full_output.split('SUMMARY:')
    if len(summary_parts) > 1:
        return summary_parts[1].strip()
    return full_output.strip()



def generate_summaries(model_path: str, config_path: str, papers: List[str]) -> Union[List[str], None]:
    """
    Function: generate_summaries

    Generates summaries for a list of input texts using the specified LLM.  Handles potential errors during 
    summary generation.

    Args:
        model_path (str): The path to the LLM model.
        config_path (str): The path to the model's configuration file.
        papers (List[str]): A list of input texts (papers) to summarize.

    Returns:
        Union[List[str], None]: A list of generated summaries, or None if an error occurs.
    """
    logger = setup_logging()

    try:
        llm_chain = setup_model(model_path, config_path)

        summaries = []
        for paper in papers:
            try:
                summary_output = llm_chain.run(paper)
                summary = extract_summary(summary_output)
                summaries.append(summary)
            except Exception as e:
                logger.error(f"Error generating summary for paper: {str(e)}", exc_info=True)
                return None # Return None to indicate failure

        return summaries

    except Exception as e:
        logger.error(f"Error in generate_summaries: {str(e)}", exc_info=True)
        return None


# Example Usage (uncomment to run):
# if __name__ == "__main__":
#     model_path = "./LLM/models/Llama-3.1-8B"  # Replace with your model path
#     config_path = "./LLM/models/Llama-3.1-8B/config.json" # Replace with your config path (if applicable)

#     papers = [
#         "This is the text of the first paper...",
#         "This is the text of the second paper...",
#     ]

#     summaries = generate_summaries(model_path, config_path, papers)

#     if summaries:
#         for i, summary in enumerate(summaries):
#             print(f"Summary {i + 1}:\n{summary}\n")
#     else:
#         print("Failed to generate summaries.")