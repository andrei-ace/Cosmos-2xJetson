from transformers import T5TokenizerFast, T5EncoderModel

# Define the model name and cache directory
model_name = "google-t5/t5-11b"
cache_dir = "./checkpoints/"  # Specify the directory where the model will be cached

# Download the tokenizer and model, and cache them in the specified directory
print("Downloading tokenizer...")
tokenizer = T5TokenizerFast.from_pretrained(model_name, cache_dir=cache_dir)
print("Downloading model...")
text_encoder = T5EncoderModel.from_pretrained(model_name, cache_dir=cache_dir)

print(f"Model and tokenizer downloaded and cached in {cache_dir}")

