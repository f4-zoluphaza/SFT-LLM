
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
from datasets import load_from_disk
from trl import SFTTrainer
import os

os.environ["huggingface_token"] = "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" 
save_path = "model_llama2_original_news"

dataset = load_from_disk("/home/nata20034/workspace/SFT-NewsletterLLM/path_to_custom_dataset_original_news")

# model_name="huggingface_llama2_models"
# model_name="ainize/kobart-news"
# model_name="beomi/open-llama-2-ko-7b"
# model_name="saltlux/Ko-Llama3-Luxia-8B"
model_name="beomi/llama-2-koen-13b"


model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", use_auth_token=os.environ["huggingface_token"])
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=os.environ["huggingface_token"])

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

training_args = transformers.TrainingArguments(
            output_dir=save_path,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=1,
            optim='paged_adamw_32bit',
            save_steps=0,
            logging_steps=25,
            learning_rate=1e-5,
            weight_decay=0.001,
            # fp16=kwargs['fp16'],
            # bf16=kwargs['bf16'],
            max_grad_norm=0.3,
            # max_steps=1,
            warmup_ratio=0.03,
            group_by_length=True,
            lr_scheduler_type='cosine',
        )

from accelerate import Accelerator

accelerator = Accelerator()

trainer = SFTTrainer(
    model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=512,
    args=training_args
)

trainer.train()

model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

# export HF_HOME=/home/nata20034/.cache/huggingface/hub