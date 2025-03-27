from datasets import load_dataset

dataset = load_dataset("openwebtext", trust_remote_code=True)
dataset.save_to_disk("openwebtext2")
