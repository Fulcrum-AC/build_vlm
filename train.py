import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from transformers import Trainer, TrainingArguments
from model import LanguageConfig, VisualConfig
from data_process import ImageCaptionDataset,read_jsonl,data_collate
from torch.utils.data import random_split
from model import VLM

def train_model():

    Lconfig=LanguageConfig(pretrained=True,test_mode=False)  
    Vconfig=VisualConfig(pretrained=True, test_mode=False)  
   
    data = read_jsonl(Lconfig.text_path)
    # print(data[0])
    dataset = ImageCaptionDataset(data,Lconfig,Vconfig)
    print(f"dataset_len:{len(dataset)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vlm_model = VLM(Lconfig, Vconfig).to(device)
    print("finish loaing vlm model")

    train_args = TrainingArguments(
        output_dir="./vlm_model_output5",
        per_device_train_batch_size=40,  # 设置训练批大小
        save_strategy="steps",        # 按步数保存
        save_steps=500,               # 每500步保存一次检查点
        save_total_limit=20,           # 只保留最新的3个检查点
        logging_strategy="steps",     # 按步数记录日志
        logging_steps=10,            # 每10步记录一次日志
        bf16=True, 
          # 关键优化部分 - 学习率策略
        learning_rate=3e-4,  # LoRA微调使用较高的学习率
        num_train_epochs=8,
        weight_decay=0.01,    # 添加权重衰减防止过拟合
        
        # 线性预热 + 余弦退火
        warmup_ratio=0.02,    # 总步数的3%用于预热
        lr_scheduler_type="cosine",  # 余弦退火
        max_grad_norm=1.0,     # 梯度裁剪，稳定训练
        save_safetensors=False # 禁用Safetensors格式
    )

    trainer = Trainer(
        model=vlm_model,
        args=train_args,
        train_dataset=dataset,
        data_collator=data_collate
    )

    trainer.train()

if __name__ == "__main__":
    train_model()  # 训练模型

    

    