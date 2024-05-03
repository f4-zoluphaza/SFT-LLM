import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

question = "Write your questions Here"

model_path='beomi/open-llama-2-ko-7b'

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto')

prompt = question
prompt_template = f'''딥러닝이 뭐야?
'''
inputs = tokenizer(prompt_template, return_tensors="pt").to("cuda")

# Generate
generate_ids = model.generate(inputs.input_ids, max_length=512)
generated_text=tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


print(generated_text)
