import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import json
from torch.utils.data import Dataset
from transformers import DefaultDataCollator
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM,SiglipImageProcessor,SiglipVisionModel,Qwen2ForCausalLM

def read_jsonl(jsonl_path):
    with open(jsonl_path, 'r', encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return data


class ImageCaptionDataset(Dataset):

    def __init__(self,data, Lconfig,Vconfig):
        super().__init__()
        self.data=data
        self.Lconfig=Lconfig
        self.Vconfig=Vconfig
        self.image_processor = SiglipImageProcessor.from_pretrained(Vconfig.model_path, cache_dir=Vconfig.cache_dir) # 得到图像的RGB张量
        self.text_tokenizer = AutoTokenizer.from_pretrained(Lconfig.model_path, cache_dir=Lconfig.cache_dir) 

    
    def format_conversation(self,conversations):
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
                formatted.append(f"<|im_start|>{role}{content}<|endoftext|><|im_end|>")
        
        # 添加结束标记并返回拼接文本
        return "".join(formatted)

    def __getitem__(self, index):
        item=self.data[index]
        if "conversations" in item:
            out_of_tokenizer=self.text_tokenizer(
                self.format_conversation(item["conversations"]),
                padding="max_length", 
                truncation=True, 
                padding_side="left",  # 设置为左填充
                max_length=32,  # 根据实际需求调整
                return_tensors="pt"
            )

        if "image" in item:
            image_path = self.Vconfig.image_path_prefix+'/'+item["image"]
            image = Image.open(image_path).convert("RGB")
            pixel_values = self.image_processor(images=image, return_tensors="pt")["pixel_values"][0]
            # print("pixel_values shape:", pixel_values.shape)
        
        return out_of_tokenizer["input_ids"], pixel_values,out_of_tokenizer["attention_mask"],out_of_tokenizer["input_ids"]
        
    def __len__(self):
        return len(self.data)


def data_collate(batch):
    input_ids=[]
    pixel_values=[]
    attention_mask=[]
    labels=[]
    for b in batch:
        one_input_ids,one_pixel_values,one_attention_mask,one_labels=b
        input_ids.append(one_input_ids)
        pixel_values.append(one_pixel_values)
        attention_mask.append(one_attention_mask)
        labels.append(one_labels)
    return{
        "input_ids": input_ids,  
        "pixel_values": pixel_values,  # 返回对话内容
        "attention_mask": attention_mask,# 返回注意力掩码
        "labels": labels  # 添加伪标签，值不重要
    }

