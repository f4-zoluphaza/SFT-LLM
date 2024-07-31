import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

os.environ["huggingface_token"] = "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" 

question = '''
누가 (Who): LG 트윈스와 롯데 자이언츠

언제 (When): 2024년 6월 14일

어디서 (Where): 서울 잠실야구장

무엇을 (What):
1. LG 트윈스가 롯데 자이언츠와의 경기에서 5대 3으로 승리하며 4연패를 끊었다.
2. LG의 선발 엔스는 4회초 솔로 홈런을 허용했지만, 퀄리티스타트를 기록하며 좋은 투구를 보였다.
3. LG의 김진성이 7회 무사 1, 2루 위기를 실점 없이 막아내며 팀 승리에 기여했다.
4. 유영찬은 8회 1사 1, 3루 위기에서 조기 투입되어 연속 삼진으로 위기를 넘겼고, 9회초 3자범퇴로 경기를 마무리했다.

어떻게 (How):
1. LG는 1회초 먼저 2실점했으나, 2회말 대거 4득점을 하며 경기를 뒤집었다.
2,. LG의 불펜진이 롯데 타선을 효과적으로 막아내며 추가 실점을 허용하지 않았다.
3. 7회 김진성의 무사 1, 2루 위기 상황에서의 무실점 투구와 8회 유영찬의 연속 삼진으로 위기를 극복했다.
4. 9회초 문보경의 쐐기포로 2점차 리드를 확립하며 승리를 굳혔다.

왜 (Why):
1. LG는 최근 4연패를 끊고 분위기 반전을 노리기 위해 승리가 필요했다.
2. 롯데는 선발 투수 이민석이 초반에 무너졌으나, 중간 계투진이 LG 타선을 잘 막으며 반격을 시도했지만 역부족이었다.

감독과 선수의 인터뷰:
1. 유영찬: "다음 이닝을 잘 막아야 한다는 생각만 하고 있었다. 경기에 집중하느라 팬들의 환호를 못 들었다."
2. 유영찬: "문보경의 쐐기포 덕분에 마음이 더 편해졌다."
3. 염경엽 감독: "유영찬이 더 강한 책임감을 가지고 성장하고 있다."
4. 유영찬: "롯데전에서 불안했던 기억이 있지만, 오늘 이후로는 그런 불안함을 잊고 싶다. 박동원 형 미트만 보고 자신 있게 던졌다."
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