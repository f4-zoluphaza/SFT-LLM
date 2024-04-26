import jsonlines
from datasets import Dataset
import warnings
warnings.filterwarnings('ignore')

dataset = []
with jsonlines.open("./custom.jsonl") as f:
    for line in f.iter():
      dataset.append(f'<s>### Instruction: \n{line["inputs"]} \n\n### Response: \n{line["response"]}</s>')

# 데이터셋 확인
print('데이터셋 확인')
print(dataset[:5])

# 데이터셋 생성 및 저장
dataset = Dataset.from_dict({"text": dataset})
dataset.save_to_disk('your_path')

# 데이터셋 info 확인
print('데이터셋 info 확인')
print(dataset)