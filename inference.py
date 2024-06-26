import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

os.environ["huggingface_token"] = "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" 

question = '''
누가 (Who): 한화 이글스와 롯데 자이언츠

언제 (When): 2024년 5월 28일

어디서 (Where): 대전 한화생명이글스파크

무엇을 (What):
1. 한화 이글스가 롯데 자이언츠와 주중시리즈 1차전을 치름.
2. 최원호 전 감독이 팀을 떠나고 정경배 수석코치가 감독대행으로 취재진과 첫 만남을 가짐.
3. 최원호 전 감독은 구단 관계자 및 선수단과 작별 인사를 나눈 후 팀을 떠남.
4. 정경배 감독대행은 선수들에게 최선을 다할 것을 당부하며 팀을 이끌어나갈 것을 다짐.

어떻게 (How):
1. 정경배 감독대행은 경기에 앞서 눈이 부어 선글라스를 벗을 수 없다고 언급하며 브리핑을 시작.
2. 최원호 전 감독과의 긴 인연을 강조하며, 그의 떠남에 대한 안타까움과 죄송함을 표현.
3. 감독 대행으로서 팀의 기조를 유지하며 코치들과 상의하여 팀을 이끌겠다고 발표.
4. 투타 고참 선수들에게 어린 선수들을 잘 이끌어달라고 부탁하며 팀의 안정성을 강조.

왜 (Why):
1. 한화는 최근 4승 1패의 성적을 기록하며 상승세를 타고 있었으나, 예상치 못한 감독 교체로 충격을 받음.
2. 팀의 기조를 유지하며 선수들의 동요를 막고 좋은 성적을 이어가기 위해 정경배 감독대행이 취임.
3. 새로운 감독이 오기 전까지 팀의 안정성을 유지하고자 함.

감독과 선수의 인터뷰:
1. 정경배 감독대행: "감독님 인사 끝나고, 선수들에겐 동요하지 말고 최선을 다하라고 얘기했다. 어쨌든 야구를 해야 하는 상황이다."
2. 정경배 감독대행: "죄송하다는 말씀 외엔 (최원호) 감독님께 드릴 말씀이 없었다. 내가 더 잘해서 도움이 됐어야 한다."
3, 정경배 감독대행: "지금 따로 수석코치를 임명할 상황도 아니고, 따로 정해놓은 것도 없다. 그냥 경기하면서 파트별 코치들과 상의하겠다."
4. 정경배 감독대행: "새로운 감독님에 대해서는 전혀 모른다. 새 감독님이 오시기 전까지 선수들이 동요하지 않도록 잘 이끌겠다."
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
>>>>>>> 8270803e50f2cef31e064ae0b3ee7c31a69c7e5d
You must generate answers according to the following rules.
1. Make a new sentence instead of using the sentence in the news provided.
2. Write the sentences based on the news content provided.
3. The answer should be no more than 700 characters.
Remember to write the sentences you make based on the news content provided.
<<<<<<< HEAD
Never make up new content.

=======
You must only generate new information based on the contents of the provided news, and not invent anything arbitrarily.
Never make up new content.
>>>>>>> 8270803e50f2cef31e064ae0b3ee7c31a69c7e5d
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
>>>>>>> 8270803e50f2cef31e064ae0b3ee7c31a69c7e5d


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