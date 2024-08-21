import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

os.environ["huggingface_token"] = "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" 

question = '''
누가 (Who): 한화 이글스와 두산 베어스

언제 (When): 2024년 6월 12일

어디서 (Where): 서울 잠실구장

무엇을 (What):
1. 한화 이글스가 두산 베어스와의 경기에서 4-3으로 승리.
2. 김경문 감독이 9회초 1사 1, 3루 상황에서 대타 문현빈에게 스퀴즈번트를 지시하여 결승점을 뽑음.
3. 김경문 감독이 경기 후 문현빈과 하이파이브를 하며 승리를 축하.

어떻게 (How):
1. 한화는 3-3으로 맞선 9회초, 1사 1, 3루 상황에서 문현빈이 스퀴즈번트로 결승점을 기록.
2. 김경문 감독이 강공 위주에서 벗어나 스퀴즈번트를 선택, 이 작전이 주효하여 승리를 거둠.
3. 두산 이승엽 감독은 스퀴즈 작전에 대비했으나 100% 스퀴즈는 예상치 못했다고 언급.
4. 김경문 감독은 연장을 피하고자 9회에 승부를 내기 위해 스퀴즈 작전을 선택했다고 밝힘.

왜 (Why):
1. 김경문 감독은 연장전이 불펜 소모를 크게 하고 이후 경기에도 나쁜 영향을 끼칠 수 있어 연장을 피하고자 했음.
2. 한화는 시즌 중반 6위 NC 다이노스와 5위 SSG 랜더스를 추격하며 순위 상승을 노리고 있음.
3. 김경문 감독은 팬들을 위해 한 경기, 한 경기 최선을 다하겠다는 의지를 보이며 팀의 분발을 촉구.
'''

# model_path='beomi/open-llama-2-ko-7b'
# model_path='./path_to_save_model'
# model_path = 'beomi/llama-2-koen-13b'
# model_path = 'path_to_save_model_json204_epochs3_llama2_13b'
model_path = 'path_to_save_model_5W1H_epochs3_llama2_13b'

model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", use_auth_token=os.environ["huggingface_token"])
tokenizer = AutoTokenizer.from_pretrained(model_path, use_auth_token=os.environ["huggingface_token"])
tokenizer.pad_token = tokenizer.eos_token


instruction = '''You are a newsletter writer who creates newsletters.
You must generate answers according to the following rules.
1. Make a new sentence instead of using the sentence in the news provided.
2. Write the sentences based on the news content provided.
3. The answer should be no more than 700 characters.
Remember to write the sentences you make based on the news content provided.
You must only generate new information based on the contents of the provided news, and not invent anything arbitrarily.
Never make up new content.
'''

prompt_template = f'''
뉴스기사 데이터: {question} instruction: {instruction} 
'''

prompt_template = f'''
instruction: {instruction}  뉴스기사 데이터: {question}
'''



# prompt_template = f'''
# ###지시 : {instruction}\n\n  ### 뉴스기사: {question} \n\n ### 답변:
# '''


# prompt_template = f'''
# 뉴스기사 데이터: {question}
# '''

inputs = tokenizer(prompt_template, return_tensors="pt").to("cuda")


# Generate
generate_ids = model.generate(inputs.input_ids, max_length=4096, repetition_penalty = 1)
generated_text=tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

def remove_before_first_star_and_after_fifth_star(text):
  # 첫 번째와 다섯 번째 *의 위치를 저장할 변수
  first_star_index = -1
  fifth_star_index = -1
  star_count = 0

  # 첫 번째와 다섯 번째 *의 위치 찾기
  for index, char in enumerate(text):
    if char == "*":
      star_count += 1
      if star_count == 1:
        first_star_index = index
      elif star_count == 5:
        fifth_star_index = index
        break

  # 첫 번째 *가 없는 경우 원래 문자열 반환
  if first_star_index == -1:
    return text

  # 다섯 번째 *가 없는 경우 첫 번째 * 이후의 문자열 반환
  if fifth_star_index == -1:
    return text[first_star_index:]

  # 첫 번째 * 이후와 다섯 번째 * 이전의 문자열 반환
  return text[first_star_index:fifth_star_index]

generated_text = remove_before_first_star_and_after_fifth_star(generated_text)
print(generated_text)

# export HF_HOME=/media1/models/