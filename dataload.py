import jsonlines
from datasets import Dataset
import warnings
warnings.filterwarnings('ignore')

dataset = []
with jsonlines.open("/home/nata20034/workspace/SFT-NewsletterLLM/json/newletter700_news1.jsonl") as f:
    for line in f.iter():
      dataset.append(f'<s>### Instruction: \n{line["inputs"]} \n\n### Response: \n{line["response"]}</s>')

# 데이터셋 확인
print('데이터셋 확인')
print(dataset[:5])

# 데이터셋 생성 및 저장
dataset = Dataset.from_dict({"text": dataset})
dataset.save_to_disk('path_to_custom_dataset_original_news')

# 데이터셋 info 확인
print('데이터셋 info 확인')
print(dataset)