from huggingface_hub import snapshot_download
import os

username = "ielhaime"
models = [
    "pszemraj/led-large-book-summary",
    "meta-llama/Llama-3.1-8B",  
    # Add more models here...
]

base_dir = f"/nobackup/{username}/models"  # Change this to your desired base directory

# Create the base directory if it doesn't exist
os.makedirs(base_dir, exist_ok=True)

for model_name in models:
    model_dir = os.path.join(base_dir, model_name.split("/")[-1])  # Extract model name from repo ID
    
    # Create the model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Download model
    snapshot_download(repo_id=model_name, local_dir=model_dir, cache_dir=model_dir)
    print(f"Downloaded {model_name} to {model_dir}")