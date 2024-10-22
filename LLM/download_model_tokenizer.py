import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from pathlib import Path

def download_and_save_model(model_name, save_dir):
    """
    Download and save a HuggingFace model and tokenizer locally.
    
    Args:
        model_name (str): Name of the model on HuggingFace
        save_dir (str): Directory to save the model and tokenizer
    """
    # Create the save directory if it doesn't exist
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading model and tokenizer from {model_name}...")
    print(f"This may take a while depending on your internet connection...")
    
    try:
        # Download and save the tokenizer
        print("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(save_dir)
        print(f"Tokenizer saved to {save_dir}")
        
        # Download and save the model
        print("Downloading model...")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model.save_pretrained(save_dir)
        print(f"Model saved to {save_dir}")
        
        print("Download completed successfully!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    MODEL_NAME = "pszemraj/led-large-book-summary"
    USERNAME = "ielhaime"
    SAVE_DIR = "/nobackup/" + USERNAME + "/model"
   
    
    # Download model and tokenizer
    download_and_save_model(MODEL_NAME, SAVE_DIR)
    
    # Verify the files were saved
    if os.path.exists(os.path.join(SAVE_DIR, "pytorch_model.bin")) and \
       os.path.exists(os.path.join(SAVE_DIR, "config.json")) and \
       os.path.exists(os.path.join(SAVE_DIR, "tokenizer.json")):
        print("\nVerification successful: All necessary files have been saved.")
        print(f"Model and tokenizer are ready to use from: {SAVE_DIR}")
    else:
        print("\nWarning: Some files may be missing. Please check the output directory.")