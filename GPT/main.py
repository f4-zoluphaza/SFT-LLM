import json
from fiveW1H_generative_GPT import main as fiveW1H_generative
from newsletter_generative_GPT import main as newsletter_generative

# 육하원칙 데이터셋 저장할 이름
new_fiveW1H_json_name = "test_fiveW1H_0925.jsonl"

# 잘못된 실험 육하원칙 뉴스레터 json 파일 경로
original_fiveW1H_json_path = '/home/nata20034/workspace/SFT-NewsletterLLM/json/5W1H_news.jsonl'

# 잘못된 실험 육하원칙 json 가져와서 gpt한테 넣기
with open(original_fiveW1H_json_path, 'r', encoding='utf-8') as file, open(f'../json/{new_fiveW1H_json_name}', 'a', encoding='utf-8') as new_file:
    for index, line in enumerate(file):
        # 각 줄을 JSON 객체로 변환
        data = json.loads(line.strip())
        # data['inputs']으로 inputs 값에 접근할 수 있음.
        print(f"잘못된 육하원칙 {index}: {data['inputs']}")
        newsletter_result = newsletter_generative(data['inputs'])
        print(f"gpt생성 {index}: {newsletter_result}")

        # 새로운 json 생성
        json_object = {
            "inputs": data['inputs'],
            "response": newsletter_result
        }
        
        new_file.write(json.dumps(json_object, ensure_ascii=False) + '\n')


print("새로운 육하원칙 dataset input 생성 완료")
