import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

question = "Write your questions Here"

model_path='./path_to_finetuned_model'

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto')

prompt = question
prompt_template = f'''Write your Prompts here
'''
inputs = tokenizer(prompt_template, return_tensors="pt").to("cuda")

# Generate
generate_ids = model.generate(inputs.input_ids, max_length=512)
generated_text=tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


print(generated_text)
