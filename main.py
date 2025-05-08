from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torch
import os
import uuid

# 本地模型路径，如果使用modelscope下载的模型则是模型下载路径，如果使用在train.py微调后的模型则是微调模型保存路径
model_path = "Qwen2-VL/Qwen2-VL-7B-Instruct"

# 加载模型
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.float16,  # 若不支持 fp16，可改为 "auto"
    device_map="auto"
)

# 加载处理器
processor = AutoProcessor.from_pretrained(model_path)

app = Flask(__name__)
CORS(app)

@app.route('/upload', methods=['POST'])
def upload():
    text_input = request.form.get('text') or ""
    image_file = request.files.get('image')
    prompt = text_input.strip() + "?" if text_input else "请描述图片内容？"

    image_inputs, video_inputs = None, None
    temp_image_path = None  # 用于后续删除

    if image_file:
        # 使用唯一文件名，避免冲突
        temp_image_path = f"/tmp/{uuid.uuid4().hex}.jpg"
        image_file.save(temp_image_path)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": temp_image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
    else:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    try:
        # 构造输入
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to("cuda")

        # 推理（无计算图，加快速度/省内存）
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=128)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

    except Exception as e:
        output_text = f"推理出错：{str(e)}"

    finally:
        # 删除临时图像，释放资源
        if temp_image_path and os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        # 释放显存
        del inputs, generated_ids, generated_ids_trimmed
        torch.cuda.empty_cache()

    return jsonify({"received_text": output_text})


if __name__ == '__main__':
    # 正式运行不要开启 debug
    app.run(host='0.0.0.0', port=5000, debug=False)
