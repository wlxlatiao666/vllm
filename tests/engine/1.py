from transformers import AutoTokenizer

model_path = "/inspire/hdd/global_public/public_models/Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)

ids = [3555, 374, 279, 7428]
decoded_from_ids = tokenizer.decode(ids, skip_special_tokens=False)
print(decoded_from_ids)