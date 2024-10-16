import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

os.environ["huggingface_token"] = "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" 

question = '''
누가 (Who): LG 트윈스와 NC 다이노스
언제 (When): 2024년 8월 11일
어디서 (Where): 잠실야구장
무엇을 (What):
1. LG 트윈스의 구본혁이 9회초 실책으로 인해 팀을 패배 위기에 빠트렸으나, 9회말 박동원의 끝내기 안타로 팀이 역전승을 거두었다.
2. LG 트윈스가 오스틴의 솔로 홈런과 박동원의 2타점 역전 끝내기 2루타로 4-3의 승리를 기록.
3. 박동원이 경기 후 구본혁과 기쁨을 나누며 팀의 위기를 극복.
어떻게 (How):
1. 구본혁은 9회초 2사 2,3루 상황에서 김성욱의 땅볼을 처리 실패하여 두 명의 주자가 홈을 밟고 1-3으로 역전당함.
2. 9회말 오스틴이 솔로 홈런으로 1점차로 추격.
3. 박동원이 2사 1,2루에서 좌측 펜스를 맞히는 2타점 역전 끝내기 2루타를 기록하여 경기를 4-3으로 마무리.
왜 (Why):
1. LG 트윈스는 5연승에 도전 중이며, 팀의 꾸준한 라인업과 선수들의 활약으로 연승 행진을 지속.
2. 구본혁은 개인 실책에도 불구하고 팀 플레이에 기여하며 부담을 덜어냄. 
3. 함덕주와 박명근의 1군 합류로 불펜진 강화, 이는 김진성의 부담을 덜어줄 것으로 기대.
'''

model_path = '/media1/models/hub/model_llama2_5W1H_0925'
# model_path="sieun1002/newsletter_5W1H_interview"

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
instruction: {instruction}  뉴스기사 데이터: {question}
'''


inputs = tokenizer(prompt_template, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")


# Generate
generate_ids = model.generate(inputs.input_ids, max_length=1500, repetition_penalty=1.2)
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
print(title)
print(content)

# 모델 실행 후 GPU 캐시 삭제
torch.cuda.empty_cache()
# export HF_HOME=/media1/models/