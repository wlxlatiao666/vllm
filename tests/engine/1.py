from transformers import AutoTokenizer

model_path = "/inspire/hdd/global_public/public_models/Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)

ids = [220, 18, 22870]
decoded_from_ids = tokenizer.decode(ids, skip_special_tokens=False)
print(decoded_from_ids)