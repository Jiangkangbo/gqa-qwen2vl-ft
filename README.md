# 在VQA上微调，通过Flask部署后端服务与前端html进行交互

## 1.环境准备

1. 确保你的电脑上至少有一张英伟达显卡，并已安装好了CUDA环境。
2. 安装Python（版本>=3.8）以及CUDA对应版本的pytorch。
3. 安装与Qwen2-VL微调相关的第三方库，可以使用以下命令：

   ```
   pip install modelscope transformers sentencepiece accelerate datasets peft swanlab qwen-vl-utils pandas json os flask flask_cors uuid tqdm numpy pandas
   ```

## 2.数据准备

1.上GQA官网下载gqa数据集，下载Questions和Images。

2.将数据转化为我们微调所需要的格式，将pre_json.py中的文件路径改成自己的文件路径然后运行。

3.从modelscope上下载模型文件到save_model_path（更改为你的保存路径）。

```
 modelscope download --model Qwen/Qwen2-VL-7B --local_dir save_model_path
```

## 3.微调训练

将train.py中的路径改成自己相对应的路径，然后运行，开始训练。利用swanlab记录训练过程，第一次使用swanlab需要先到官网注册，第一次运行时需要登陆。swanlab相关问题可以参考：

[SwanLab快速开始](https://docs.swanlab.cn/guide_cloud/general/quick-start.html)

## 4.运行Flask后台服务

运行main.py开启后台服务，监听https请求，加载模型进行推理。

## 5.运行前端界面

用浏览器打开web.html，打开之后效果图如web.jpg所示，可以选择图片（可选），输入问题（也支持语音输入），点击提交则会通过https post方法将请求上传到Flask后台服务，后台进行推理后将结果返回web页面显示并朗读出来。

笔者对于前端了解不深，做出来的界面并不完善，还请诸位读者海涵。
