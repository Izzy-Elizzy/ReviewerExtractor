from huggingface_hub import snapshot_download
import os

username = "ielhaime"
model_name = "pszemraj/led-large-book-summary" 
# save_dir = f"/nobackup/{username}/model"
save_dir = f"./model"

# Create the save directory if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

# Download model
snapshot_download(repo_id=model_name, local_dir=save_dir, cache_dir=save_dir)
