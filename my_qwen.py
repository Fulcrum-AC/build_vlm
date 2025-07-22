import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
from torch import nn
from typing import Optional, Tuple, Union
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM, BaseModelOutputWithPast
from transformers.utils import add_start_docstrings_to_model_forward, can_return_tuple, auto_docstring
from transformers import AutoConfig
import torch.nn as nn

class MyQwenModel(Qwen2ForCausalLM):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

    @auto_docstring
    def forward(
        self,
        feature_proj,
        input_ids: Optional[torch.LongTensor] = None,
        image_features: Optional[torch.FloatTensor] = None,  
        attention_mask: Optional[torch.Tensor] = None,
        text_tokenizer: Optional[torch.nn.Module] = None,
        **kwargs
    ) -> CausalLMOutputWithPast:
        """
        扩展的 Qwen2 前向传播方法，支持视觉语言建模
        
        Args:
            input_ids (torch.Tensor, optional): 
                输入token IDs, 形状 [batch_size, sequence_length]
            aligned_image_features (torch.Tensor, optional):
                对齐的图像分块特征张量，形状为 [batch_size, num_patches, model_dim]
            **kwargs: 其他传递给父类Qwen2ForCausalLM.forward的参数
        
        Returns:
            transformers.modeling_outputs.CausalLMOutputWithPast: 语言模型输出
        """

        # 将图像特征与文本对齐
        aligned_image_features = feature_proj(image_features).to(self.device)

        image_start_token_id = text_tokenizer.convert_tokens_to_ids("<|vision_start|>")

        # 处理图像和文本的融合输入
        combined_embeds,new_attention_mask = self.build_fused_embeddings(
            input_ids, 
            aligned_image_features, 
            image_start_token_id=image_start_token_id,
            attention_mask=attention_mask
        )
        # print(f"combined_embeds: {combined_embeds}")  
        # print(f"new_attention_mask: {new_attention_mask}")
        self.new_attention_mask=new_attention_mask
        
        labels = self.build_labels(input_ids, text_tokenizer)

        # print(f"labels:{labels}")
        # print(f"labels shape:{labels.shape}")

        # 生成自定义因果掩码，并结合处理后的填充掩码得到融合掩码
        combined_mask = self.build_custom_attention_mask(
            input_ids, 
            combined_embeds.size(1),
            text_tokenizer,
            device=combined_embeds.device
        )

        # 用融合后的嵌入代替原始input_ids
        return super().forward(
            input_ids=None,  # 禁用input_ids
            inputs_embeds=combined_embeds,  # 使用自定义嵌入
            attention_mask=combined_mask,  # 使用新的注意力掩码
            # attention_mask=new_attention_mask,
            labels=labels # 保留标签参数以计算损失
        )
    
    def build_custom_attention_mask(self, input_ids, new_seq_len, tokenizer, device):
        """构建自定义注意力掩码，结合填充掩码和因果掩码"""
        batch_size = input_ids.size(0)
        
        # 1. 创建基础掩码 (batch_size, 1, new_seq_len, new_seq_len)
        mask = torch.ones((batch_size, 1, new_seq_len, new_seq_len), device=device)
        
        # 找到特殊token的位置
        assistant_token_id = tokenizer.convert_tokens_to_ids("assistant")
        image_token_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
        
        for b in range(batch_size):
            # 找到assistant位置
            assistant_pos = (input_ids[b] == assistant_token_id).nonzero(as_tuple=True)[0]
            if len(assistant_pos) == 0:
                continue
            answer_start = assistant_pos[0].item()
            
            # 调整answer_start位置（考虑图像特征扩展）
            image_positions = (input_ids[b] == image_token_id).nonzero(as_tuple=True)[0]
            for pos in image_positions:
                if pos < answer_start:
                    answer_start += 196
            
            # 2. 设置因果注意力模式
            # 问题+图像部分可以互相看见
            mask[b, 0, :answer_start+1, :answer_start+1] = 1
            # 答案部分只能看到之前的内容
            mask[b, 0, answer_start+1:, answer_start+1:].tril_()
            # 问题和图像部分不能看到答案部分
            mask[b, 0, :answer_start+1, answer_start+1:] = 0
        
        # 3. 结合填充掩码
        # new_attention_mask 形状是 (batch_size, new_seq_len)
        if self.new_attention_mask is not None:
            # 扩展填充掩码到4D
            padding_mask = self.new_attention_mask.unsqueeze(1).unsqueeze(2)
            # 广播填充掩码
            mask = mask.bool() & padding_mask
        
        return mask.bool()

    def build_labels(self, input_ids, tokenizer):
        """
        构建图像问答任务的标签序列
        
        Args:
            input_ids (torch.Tensor): 原始文本token序列 [batch_size, seq_len]
            tokenizer: 文本tokenizer
            
        Returns:
            torch.Tensor: 标签序列 [batch_size, seq_len + 196 * num_images]
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # 获取特殊token的ID
        vision_start_token_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
        vision_end_token_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")
        assistant_token_id = tokenizer.convert_tokens_to_ids("assistant")
        eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        im_start_token=tokenizer.convert_tokens_to_ids("<|im_start|>")
        im_end_token=tokenizer.convert_tokens_to_ids("<|im_end|>")

        # print(f"assistant_token_id:{assistant_token_id}")
        # print(f"eos_token_if:{eos_token_id}")

        # 初始化标签矩阵 (全为-100)
        labels = torch.full((batch_size, seq_len), -100, dtype=torch.long, device=device)
        in_assistant_response = False
        #======================================================================================================================
        #======================================================================================================================
        #======================================================================================================================
        for b in range(batch_size):
            idx = 0
            while idx < seq_len:
                if input_ids[b, idx] == assistant_token_id:
                   labels[b, idx] = -100  # assistant标记本身不预测
                   in_assistant_response = True
                   idx+=1
                   continue

                if input_ids[b, idx] == eos_token_id and in_assistant_response:
                    labels[b, idx] = eos_token_id  # eos标记需要预测
                    in_assistant_response = False

                if in_assistant_response:
                    labels[b, idx] = input_ids[b, idx]

                idx+=1

          # 在每个序列前面添加196个-100
        add = torch.full((batch_size, 196), -100, dtype=torch.long, device=device)
        labels = torch.cat([add, labels], dim=1)  # 拼接后形状为 (batch_size, 196 + seq_len)

        return labels
        
    #======================================================================================================================
        #======================================================================================================================
        #======================================================================================================================
    def build_fused_embeddings(
        self,
        input_ids,
        aligned_image_features,
        image_start_token_id,
        attention_mask: Optional[torch.Tensor] = None
    ):
        # print(f"attention_mask_shape: {attention_mask.shape}")
        # 获取文本嵌入
        text_embeds = self.model.embed_tokens(input_ids)
        # print(f"嵌入向量维-度（文本+图像token):{text_embeds.shape}")
        # print(f"序列长度:{text_embeds.shape[0]}")

        batch_size, seq_len, hidden_size = text_embeds.shape
        # print(f"batch_size:{batch_size}, seq_len:{seq_len}, hidden_size:{hidden_size}")
        
        new_seq_len = seq_len+196
        combined_embeds = torch.zeros((batch_size, new_seq_len, hidden_size), dtype=text_embeds.dtype, device=text_embeds.device)
        # 创建布尔类型掩码
        add_attention_mask=torch.ones(
            (batch_size,196),
            dtype=torch.bool,
            device=text_embeds.device
        )
        # print(f"add_attention_mask_shape:{add_attention_mask.shape}")
        new_attention_mask = torch.cat([attention_mask,add_attention_mask],dim=1)
        
        # print(combined_embeds.shape)

        # 图像文本拼接
        if aligned_image_features is not None:
            for b in range(batch_size):
                old_id=0
                new_id=0
                while old_id < len(text_embeds[b]):
                    combined_embeds[b,new_id]=text_embeds[b,old_id]
                    if input_ids[b,old_id]==image_start_token_id: # 若复制的内容为<|vision_start|>,则立刻添加196个图像特征
                        combined_embeds[b,new_id+1:new_id+197] = aligned_image_features[b]
                        new_id+=197
                    else:
                        new_id+=1
                    old_id+=1

        # print(f"combined_embeds形状:{combined_embeds.shape}")  # 输出融合后的嵌入形状
        # print(f"new_attention_mask形状:{new_attention_mask.shape}")  # 输出新的注意力掩码形状
        # print(f"new_attention_mask:{new_attention_mask}")  # 输出新的注意力掩码

        return combined_embeds,new_attention_mask
    
