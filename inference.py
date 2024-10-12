import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

os.environ["huggingface_token"] = "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" 

question = '''
누가 (Who): SSG 랜더스와 KIA 타이거즈
언제 (When): 2024년 6월 12일
어디서 (Where): 인천 SSG랜더스필드
무엇을 (What):
1. KIA 타이거즈가 SSG 랜더스와의 경기에서 0-5로 뒤지던 경기를 13-7로 역전승.
2. 최형우가 한 경기 최다 6타점을 기록하며 KBO리그 역대 개인 통산 최다 루타 대기록 달성.
3. 이우성의 주루 플레이와 3안타 활약이 승리에 결정적인 기여.

어떻게 (How):
1. 6회초, KIA가 2-5로 추격하는 상황에서 이우성이 안타를 치고 나가 소크라테스의 안타 때 2루까지 진루.
2. 김태군의 보내기 번트 시도 중 이우성이 3루로 뛰었으나 SSG 포수 김민식의 빠른 송구로 인해 아웃될 위기.
3. 3루심이 세이프 선언, 비디오 판독 결과도 세이프 판정 유지.
4. 이우성의 간절한 주루와 최정의 약간의 안일한 태그가 변수로 작용.
5. 이우성이 3루에 살아남으며 김태군이 우중간 1타점 적시타로 역전 신호탄.
6. KIA는 6회와 7회 연속 타자일순하며 각각 4점, 7점을 추가해 역전승.

왜 (Why):
1. KIA는 전날 연장 접전패를 설욕하기 위해 경기에서 승리를 노림.
2. 이우성의 결정적인 주루 플레이와 팀의 타자일순 공격이 역전승의 주된 요인.
3. SSG는 초반 5점 리드를 지키지 못하고 대량 실점하며 역전패.

감독과 선수의 인터뷰:
인터뷰 없음
'''

# model_path='beomi/open-llama-2-ko-7b'
# model_path='./path_to_save_model'
# model_path = 'beomi/llama-2-koen-13b'
# model_path = 'path_to_save_model_5W1H_epochs3_llama2_13b'


model_path = 'sieun1002/newsletter_5W1H_interview'
# model_path = 'path_to_save_model_5W1H_response_epochs4_'

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

# prompt_template = f'''
# 뉴스기사 데이터: {question} instruction: {instruction} 
# '''

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

# 뉴스레터 반복이 안 되게 하는 함수
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

# 제목괴 내용을 분리해주는 함수
def split_title_and_content(text):
  first_start_index = -1
  fourth_start_index = -1
  star_count = 0

  for index, char in enumerate(text):
    if char == "*":
      star_count += 1
      if star_count == 1:
        first_start_index = index 
      elif star_count == 4:
        fourth_start_index = index 
        break
  
  title = text[first_start_index:fourth_start_index+1]
  content = text[fourth_start_index+1:]
  return title, content

generated_text = remove_before_first_star_and_after_fifth_star(generated_text)
title, content = split_title_and_content(generated_text)
print("제목" + title)
print("내용" + "\n" + content)
# export HF_HOME=/media1/models/