import jsonlines
from datasets import Dataset
import warnings
import json

file_path = "./newsletter700.jsonl"

warnings.filterwarnings('ignore')

def clean_file(file_path):
    """파일을 열어 제어 문자를 제거하거나 이스케이프 처리합니다."""
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    cleaned_lines = [line.replace('\n', '\\n').replace('\r', '\\r') for line in lines]
    return cleaned_lines

def load_and_process_data(file_path):
    """데이터를 로드하고 처리합니다."""
    cleaned_lines = clean_file(file_path)
    dataset = []
    for line in cleaned_lines:
        try:
            data = json.loads(line)
            cleaned_inputs = clean_text(data["inputs"])
            cleaned_response = clean_text(data["response"])
            dataset.append(f'<s>### Instruction: \n{cleaned_inputs} \n\n### Response: \n{cleaned_response}</s>')
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
    return dataset

# 파일 처리 및 데이터셋 생성
dataset = load_and_process_data("./newsletter700.jsonl")

# 데이터셋 확인
print('데이터셋 확인')
print(dataset[:5])

# 데이터셋 생성 및 저장
dataset = Dataset.from_dict({"text": dataset})
dataset.save_to_disk('./')

# 데이터셋 info 확인
print('데이터셋 info 확인')
print(dataset)
