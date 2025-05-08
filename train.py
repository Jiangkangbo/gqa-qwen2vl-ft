import torch
from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer
from swanlab.integration.transformers import SwanLabCallback
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
)
import swanlab
import json
import os

#图像缩放大小，为了保持batch维度一致，根据自己的实际需求进行调整
resized_height = 300
resized_width = 500

prompt = "You are a visual question answer assistant. Answer the following questions based on the pictures:"
model_id = "Qwen/Qwen2-VL-7B-Instruct"  #modelscope上模型id
local_model_path = "../../../Qwen2-VL-7B-Instruct" #提前下载模型到本地的保存路径，根据自己实际路径进行修改
train_dataset_json_path = "train_balanced_questions.json" #训练数据集所用json
val_dataset_json_path = "val_balanced_questions.json" #测试数据集json
output_dir = "./output/Qwen2-VL-7B-ft"

MAX_LENGTH = 8192

# 如果没有提前将模型下载到本地则需要使用下面这行代码进行模型下载，建议提前下载到本地
# model_dir = snapshot_download(model_id, cache_dir="./", revision="master")

#加载模型
tokenizer = AutoTokenizer.from_pretrained(local_model_path, use_fast=False, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(local_model_path)

#数据预处理
def process_func(example):
    conversation = example["conversations"]
    image_file_path = conversation[0]["image"]
    output_content = conversation[1]["answer"]
    question = conversation[0]["question"]
    prompt = "You are a visual question answer assistant. Answer the following questions based on the pictures:" + question

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"{image_file_path}",
                    "resized_height": resized_height,
                    "resized_width": resized_width,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    
    instruction = dict(inputs)
    response = tokenizer(f"{output_content}", add_special_tokens=False)

    input_ids = instruction["input_ids"][0].tolist() + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"][0].tolist() + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"][0]) + response["input_ids"] + [tokenizer.pad_token_id]

    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    return {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention_mask),
        "labels": torch.tensor(labels),
        "pixel_values": instruction["pixel_values"].squeeze(0),
        "image_grid_thw": instruction["image_grid_thw"].squeeze(0),
    }

def predict(messages, model):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    generated_ids = model.generate(**inputs, max_new_tokens=MAX_LENGTH)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]

origin_model = Qwen2VLForConditionalGeneration.from_pretrained(
    local_model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
origin_model.enable_input_require_grads()

train_ds = Dataset.from_json(train_dataset_json_path)
train_dataset = train_ds.map(process_func)

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,
    r=64,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
)

train_peft_model = get_peft_model(origin_model, config)

args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=8,
    logging_steps=10,
    logging_first_step=10,
    num_train_epochs=2,
    save_steps=100,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="none",
)

# 使用swanlab记录训练过程
swanlab_callback = SwanLabCallback(
    project="Qwen2-VL-ft-gqa-7B",
    experiment_name="qwen2vl-tranin-val-balanced",
    config={
        "model": "https://modelscope.cn/models/Qwen/Qwen2-VL-7B-Instruct",
        "dataset": "gqa",
        "model_id": model_id,
        "train_dataset_json_path": train_dataset_json_path,
        "val_dataset_json_path": val_dataset_json_path,
        "output_dir": output_dir,
        "prompt": prompt,
        "train_data_number": len(train_ds),
        "token_max_length": MAX_LENGTH,
        "lora_rank": 64,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
    },
)

trainer = Trainer(
    model=train_peft_model,
    args=args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    callbacks=[swanlab_callback],
)

try:
    trainer.train()
    # 保存微调后的完整模型
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
except Exception as e:
    print("❌ 训练中发生异常：", e)

val_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=True,
    r=64,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
)

#训练完成使用最后一个checkpoint的参数来进行推理测试
load_model_path = f"{output_dir}/checkpoint-{max([int(d.split('-')[-1]) for d in os.listdir(output_dir) if d.startswith('checkpoint-')])}"
print(f"load_model_path: {load_model_path}")
val_peft_model = PeftModel.from_pretrained(origin_model, model_id=load_model_path, config=val_config)

with open(val_dataset_json_path, "r") as f:
    test_dataset = json.load(f)

test_image_list = []
right_count = 0
for item in test_dataset:
    image_file_path = item["conversations"][0]["image"]
    question = item["conversations"][0]["question"]
    label = item["conversations"][1]["answer"]
    prompt = "You are a visual question answer assistant. Answer the following questions with one word based on the pictures :" + question

    messages = [{
        "role": "user",
        "content": [
            {
            "type": "image", 
            "image": image_file_path,
            "resized_height": resized_height,
            "resized_width": resized_height
            },
            {"type": "text", "text": prompt},
        ]
    }]

    response = predict(messages, val_peft_model)

    print(f"predict:{response}")
    print(f"answer:{label}\n")

    if response.lower() == label.lower():
        right_count += 1

    test_image_list.append(swanlab.Image(image_file_path, caption=response))

print("测试数据集数量", len(test_dataset))
print("预测正确数量：", right_count)
print("正确率", right_count / len(test_dataset))

swanlab.log({"Prediction": test_image_list})
swanlab.finish()
