#本代码是为了将下载的gqa数据集中的json文件转化为我们微调所需要的数据集json文件

from tqdm import tqdm
import json
import numpy as np
import pandas as pd

# 保存转化后的json路径
save_path = "your_path_to_save_json.json"

#下载gqa Images的保存路径
datasets_images = "/mnt/sda/song/datasets/gqa/images/"

#路径为下载gqa数据集中questions中的json文件
with open('gqa_path/train_balanced_questions.json', 'r') as f:
    data = json.load(f)
    
conversations = []
keys = list(data.keys())
datasets_size = len(keys)
for i in tqdm(range(datasets_size)):
    key = keys[i]
    imageId = data[key]['imageId']
    question = data[key]['question']
    answer = data[key]['answer']
    fullAnswer = data[key]['fullAnswer']

    image_path = datasets_images + imageId + ".jpg"
    conversations.append({
        "id": key,
        "conversations": [
            {
                "role": "user",
                "image": image_path,
                "question": question
            },
            {
                "role": "assistant", 
                "answer": answer
            }
        ]
    })

with open(save_path, 'w', encoding='utf-8') as f:
    json.dump(conversations, f, ensure_ascii=False, indent=2)
