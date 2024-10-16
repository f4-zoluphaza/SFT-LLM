import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

os.environ["huggingface_token"] = "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" 

question = '''
누가 (Who): KIA 타이거즈와 키움 히어로즈

언제 (When): 2024년 8월 15일

어디서 (Where): 서울 고척스카이돔

무엇을 (What):
1. KIA 타이거즈가 키움 히어로즈와의 경기에서 12대1로 승리하며 주중 3연전을 위닝 시리즈로 장식.
2. KIA의 김도영, KBO리그 역대 최연소 및 최소경기 30홈런-30도루 기록 달성.
3. KIA 선발 양현종, 7이닝 1실점의 퀄리티스타트 플러스 투구를 펼침.

어떻게 (How):
1. KIA는 2회초 이창진의 희생플라이로 선취점을 얻고, 3회말에는 키움 송성문이 동점 솔로홈런으로 응수.
2. 4회초, KIA 김태군이 투런 홈런을 쳐 3-1로 리드를 잡음.
3. 5회초, 김도영의 투런 홈런으로 격차를 5-1로 벌림.
4. 7회초에는 나성범이 2타점 적시타를 포함하여 추가 점수를 올림.
5. 8회초, 나성범의 스리런 홈런으로 12-1로 승부에 쐐기를 박음.

왜 (Why):
1. KIA는 키움을 상대로 연승을 거두며 시즌 65승을 기록, 승률을 높이고 순위 경쟁에서 좋은 위치를 유지하기 위함.
2. 키움은 초반에 호투하던 선발 투수 헤이수스가 무너지면서 공격에서도 KIA 투수진을 공략하지 못해 패배. 시즌전적 49승62패로 하락.
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