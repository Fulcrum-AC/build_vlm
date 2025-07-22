import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import torch
from PIL import Image
import json
from tqdm import tqdm
from model import VLM, LanguageConfig, VisualConfig
from data_process import ImageCaptionDataset, read_jsonl
from transformers import SiglipImageProcessor
from transformers import AutoTokenizer

def format_conversation(conversations):
        """将对话历史格式化为模型输入文本"""
        formatted = []
        for turn in conversations:
            role = turn["role"]
            content = turn["content"]
            
            # 处理图像标记
            if '<image>' in content:
                content = content.replace('<image>', '<|vision_start|><|vision_end|>')
            
            if role == "user":
                formatted.append(f"<|im_start|>{role}{content}<|endoftext|><|im_end|>")
            if role == "assistant":
                formatted.append(f"<|im_start|>{role}{content}|endoftext|><|im_end|>")
        
        # 添加结束标记并返回拼接文本
        return "".join(formatted)


def generate_complete_response(
    model,
    tokenizer,
    image_processor,
    image_path,
    question,
    max_new_tokens=64,
    temperature=0.7,
    top_p=0.9,
    # repetition_penalty=1.2,  # 添加重复惩罚参数
    # no_repeat_ngram_size=3   # 禁止重复的n-gram大小
):
    # 确保结束符被正确添加到词表
    device = model.device  # 获取模型所在设备
    
    # 处理图像
    image = Image.open(image_path).convert("RGB")
    pixel_values = image_processor(images=image, return_tensors="pt")["pixel_values"].to(device)  # 直接移到设备
    
    # 构建对话格式
    conversation = [
        {"role": "user", "content": f"{question}<image>"}
    ]
    formatted = format_conversation(conversation)  # 使用独立函数而非模型方法
    formatted+="<|im_start|>assistant"

    # 分词（不截断）
    input_dict = tokenizer(
        formatted,
        return_tensors="pt",
        padding=True,
        truncation=False,
        return_length=True
    )
    input_ids = input_dict["input_ids"].to(device)  # 直接移到设备
    attention_mask = input_dict["attention_mask"].to(device)  # 直接移到设备
    
    # 准备模型输入
    inputs = {
        "input_ids": [input_ids],
        "pixel_values": [pixel_values[0]],  # 保持维度一致
        "attention_mask": [attention_mask],
        "labels": [input_ids]  # 伪标签
    }
    
    # 存储完整生成的token IDs
    all_generated_ids = input_ids.clone().detach()
    
    # 自回归生成循环
    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(**inputs)
        
        # 获取最后一个token的logits
        next_token_logits = outputs.logits[:, -1, :]

        #  # 1. 应用重复惩罚
        # if repetition_penalty != 1.0:
        #     # 获取已生成的所有token
        #     prev_tokens = all_generated_ids[0].tolist()
        #     # 对已出现过的token施加惩罚（降低其概率）
        #     for token in set(prev_tokens):
        #         next_token_logits[0, token] /= repetition_penalty
        
        # # 2. 检查n-gram重复
        # if no_repeat_ngram_size > 0 and len(all_generated_ids[0]) >= no_repeat_ngram_size:
        #     # 获取最后 n-1 个token
        #     prev_ngram = all_generated_ids[0, -(no_repeat_ngram_size-1):].tolist()
        #     banned_tokens = []
            
        #     # 在已生成序列中寻找相同的n-gram前缀
        #     for i in range(len(all_generated_ids[0]) - no_repeat_ngram_size + 1):
        #         if prev_ngram == all_generated_ids[0, i:i+no_repeat_ngram_size-1].tolist():
        #             # 将紧跟在相同前缀后的token加入禁用列表
        #             banned_tokens.append(all_generated_ids[0, i+no_repeat_ngram_size-1].item())
            
        #     # 将被禁用的token的概率设为0
        #     next_token_logits[0, banned_tokens] = float('-inf')

        # 应用温度采样
        next_token_logits = next_token_logits / temperature
        
        # 应用top-p采样
        next_token_probs = torch.softmax(next_token_logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(next_token_probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
        next_token_probs[indices_to_remove] = 0
        
        # 采样下一个token
        next_token = torch.multinomial(next_token_probs, num_samples=1)
        
        # 检查是否生成了结束符
        if next_token.item() == tokenizer.convert_tokens_to_ids("<|endoftext|>"):
            break
        # 更新输入 (确保所有张量都在相同设备)
        all_generated_ids = torch.cat([all_generated_ids, next_token], dim=-1)
        new_attention_mask = torch.ones_like(all_generated_ids)

        # # 检查连续重复
        # if len(all_generated_ids[0]) > 10:  # 至少生成10个token后才检查
        #     last_tokens = all_generated_ids[0, -10:].tolist()
        #     if len(set(last_tokens)) <= 2:  # 如果最近10个token中只有1-2个不同的token
        #         print("检测到重复生成，提前停止")
        #         break

        inputs = {
            "input_ids": [all_generated_ids],
            "pixel_values": [pixel_values[0]],  # 使用原始图像特征
            "attention_mask": [new_attention_mask],
            "labels": [all_generated_ids]
        }
    
    # 解码完整序列
    full_response = tokenizer.decode(
        all_generated_ids[0],
        skip_special_tokens=False
    )
    
    # 提取助理回答部分
    start_token = "assistant"
    end_token = "<|im_end|>"
    
    start_idx = full_response.find(start_token)
    if start_idx != -1:
        start_idx += len(start_token)
        end_idx = full_response.find(end_token, start_idx)
        if end_idx == -1:
            return full_response[start_idx:].strip()
        return full_response[start_idx:end_idx].strip()
    
    return full_response

def post_process_response(response):
    """后处理生成的回答，去除重复内容和乱码"""
    # 1. 移除连续重复的句子
    sentences = response.split('。')
    # 2. 过滤非中文字符
    import re
    filtered_response = sentences[0]
    chinese_only = re.sub(r'[^\u4e00-\u9fff，。！？、]', '', filtered_response)
    
    return chinese_only

if __name__ == "__main__":
    Lconfig = LanguageConfig(pretrained=False, test_mode=True)
    Vconfig = VisualConfig(pretrained=False, test_mode=True)
    text_tokenizer = AutoTokenizer.from_pretrained(Lconfig.model_path, cache_dir=Lconfig.cache_dir)
    # print(f"default eos token id:{text_tokenizer.eos_token_id}")
    # print("\n特殊Token映射字典:")

    # 加载数据集
    data = read_jsonl(Lconfig.text_path)
    dataset = ImageCaptionDataset(data, Lconfig, Vconfig)
    
    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VLM(Lconfig, Vconfig).to(device)

    checkpoint_path="./vlm_model_output5/checkpoint-39000/pytorch_model.bin"
    # checkpoint_path="./checkpoint-26000/pytorch_model.bin"
    # 加载检查点
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)
    model.eval()

    response = generate_complete_response(
        model=model,
        tokenizer=model.text_tokenizer,
        image_processor=model.image_processor,
        image_path="/data/build_vlm/datasets/try_images/train-00000-of-00001_image_210_0.jpg",
        question="用简洁的语言描述这张图片",
        max_new_tokens=200,
        temperature=0.8,
        top_p=0.5
    )
    # response = post_process_response(response)
    print(response)