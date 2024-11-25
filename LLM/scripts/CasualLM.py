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
import TextAnalysis as TA

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
    """Sets up the model, now model-agnostic."""
    logger = logging.getLogger(__name__)
    try:
        # Load configuration (model-agnostic)
        if config_path:
            config = AutoConfig.from_pretrained(config_path)
            logger.info(f"Loaded model configuration from {config_path}")
        else:
            config = AutoConfig.from_pretrained(model_path)
            logger.warning("No configuration file provided. Loading from model path.")

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )

        # Device mapping
        if torch.cuda.is_available():
            device = torch.device("cuda:0")  # Use first available CUDA device
            logger.info(f"Moving model to {device}")
            model.to(device)
        else:
            device = torch.device("cpu")
            logger.info("Using CPU")


        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            max_new_tokens=8192,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            device=device  # Explicitly set device for pipeline
        )
        
        llm = HuggingFacePipeline(pipeline=pipe, model_kwargs={'temperature': 0.5})

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


def extract_summary(llm_chain_output):
    """Extract the summary from the LLM chain output."""
    full_output = llm_chain_output.get('text', '')
    summary_parts = full_output.split('SUMMARY:')
    if len(summary_parts) > 1:
        return summary_parts[1].strip()
    return full_output.strip()



def generate_summaries(model_path, config_path, papers):
    """Generates summaries."""
    logger = setup_logging()

    try:

        llm_chain = setup_model(model_path, config_path)  # Pass config_path

        summaries = []
        for paper in papers:
            try:
                summary_output = llm_chain.run(paper)
                summary = extract_summary(summary_output)
                summaries.append(summary)
            except Exception as e:
                logger.error(f"Error generating summary: {str(e)}", exc_info=True)
                return None

        return summaries

    except Exception as e:
        logger.error(f"Error in generate_summaries: {str(e)}", exc_info=True)
        return None




# if __name__ == "__main__":
#     model_path = "./LLM/models/Llama-3.1-8B"  # Or path to your desired model
#     config_path = "./LLM/models/Llama-3.1-8B" # Or None, or your config path
    


#     papers = [
#         "The first paper text goes here...",
#         "The second paper text goes here...",
#         # ... more papers
#     ]

#     summaries = generate_summaries(model_path, config_path, papers)

#     if summaries:
#         for i, summary in enumerate(summaries):
#             print(f"Summary {i + 1}:\n{summary}\n")
#     else:
#         print("Failed to generate summaries.")