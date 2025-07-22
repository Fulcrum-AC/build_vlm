import torch
from dataclasses import dataclass
from transformers.modeling_outputs import CausalLMOutputWithPast
from visual_model import VisualModel
import torch.nn as nn
from typing import Optional
from my_qwen import MyQwenModel
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM,SiglipImageProcessor,SiglipVisionModel,Qwen2ForCausalLM
from peft import LoraConfig, get_peft_model
from typing import List, Optional

@dataclass
class LanguageConfig():
    pretrained: Optional[bool] = True
    test_mode: Optional[bool] = False
    model_path: str = "Qwen/Qwen2.5-0.5B"
    torch_dtype: torch.dtype = torch.bfloat16
    cache_dir: str = "/data/build_vlm/cache"
    trust_remote_code: bool = True
    
    @property
    def text_path(self) -> str:
        """动态计算text_path"""
        if self.test_mode:
            return "/data/build_vlm/datasets/test_data.jsonl"
        else:
            return (
                "/data/build_vlm/datasets/pretrain_vlm_data.jsonl" 
                if self.pretrained else 
                "/data/build_vlm/datasets/sft_vlm_data.jsonl"
            )
        
@dataclass
class VisualConfig:
    pretrained: bool = True  # 只保留一个pretrained定义
    test_mode: bool = False
    model_path: str = "google/siglip-base-patch16-224"
    cache_dir: str = "/data/build_vlm/cache"
    
    @property
    def image_path_prefix(self) -> str:
        """根据test_mode和pretrained动态计算图像路径前缀"""
        if self.test_mode:
            return "/data/build_vlm/datasets/try_images"
        else:
            # 非测试模式下，根据pretrained选择路径
            return "/data/build_vlm/datasets/pretrain_images" if self.pretrained else "/data/build_vlm/datasets/sft_images"

@dataclass
class MultiModalConfig():
    replace_token_id: int
    image_context_length: int = 196
    image_feature_hidden_size: int = 768
    cache_dir: str = "/data/build_vlm/cache"


class VLM(nn.Module):

    def __init__(self, Lconfig: LanguageConfig, Vconfig: VisualConfig, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.image_processor = SiglipImageProcessor.from_pretrained(Vconfig.model_path, cache_dir=Vconfig.cache_dir) # 得到图像的RGB张量
        self.text_tokenizer = AutoTokenizer.from_pretrained(Lconfig.model_path, cache_dir=Lconfig.cache_dir) 
        self.LLM = MyQwenModel.from_pretrained(Lconfig.model_path, torch_dtype = Lconfig.torch_dtype, trust_remote_code = Lconfig.trust_remote_code).to(self.device)
        self.visual_encoder = VisualModel.from_pretrained(Vconfig.model_path).to(self.device,Lconfig.torch_dtype)

        Vhidden_dim = self.visual_encoder.vision_embed_dim
        Lhidden_dim = self.LLM.config.hidden_size
        self.feature_proj = nn.Sequential(
            nn.Linear(Vhidden_dim, Lhidden_dim, dtype=Lconfig.torch_dtype),
            nn.GELU(),
            nn.Linear(Lhidden_dim, Lhidden_dim, dtype=Lconfig.torch_dtype)
        )

        # 根据阶段配置参数可训练性
        self.configure_parameter_trainability(Lconfig.pretrained)

    def configure_parameter_trainability(self, is_pretrain: bool):
        """配置参数可训练性"""

        # 冻结视觉编码器
        for param in self.visual_encoder.parameters():
            param.requires_grad = False

        # 投影层始终可训练
        for param in self.feature_proj.parameters():
            param.requires_grad = True
        # 配置LoRA
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj"],
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
        )
        
        # 应用LoRA
        self.LLM = get_peft_model(self.LLM, lora_config)
        for name, param in self.LLM.named_parameters():
            if "lora_" not in name:  # 非LoRA参数
                param.requires_grad = False
            
        # 打印可训练参数
        self.LLM.print_trainable_parameters()


    def make_feature_proj(self, Vhidden_dim, Lhidden_dim, Lconfig):
        for name, module in self.feature_proj.named_children():
            if "Linear" in module._get_name(): 
                module.weight.data.normal_(mean=0.0, std = 0.01)
                module.bias.data.zero_()

    def forward(
        self,
        input_ids,
        pixel_values,
        attention_mask,
        labels,
        **kwargs
    ):
        # print("Successfully pass para")
        input_ids=torch.cat(input_ids,dim=0)
        pixel_values=torch.cat([one_pixel_value.unsqueeze(0) for one_pixel_value in pixel_values],dim=0)
        attention_mask=torch.cat(attention_mask,dim=0)
        labels=torch.cat(labels,dim=0)
        # print(f"input_ids_shape:{input_ids.shape}")
        # print(f"pixel_values_shape:{pixel_values.shape}")
        # print(f"attention_mask_shape:{attention_mask.shape}")
        # print(f"labels_shape:{labels.shape}")

        # 根据像素值提取图像特征
        with torch.no_grad():
            # 确保 image 的数据类型为 bfloat16
            pixel_values = pixel_values.to(self.device,dtype=torch.bfloat16)
            image_features = self.visual_encoder(pixel_values=pixel_values)
            image_features = image_features.detach()
        # print("Successfully get image features")
        # print("image features shape:", image_features.shape)
         
        input_ids=input_ids.to(self.device)
        # print(f"input_ids shape: {input_ids.shape}")
        # print("===========")
        # print(self.text_tokenizer.convert_tokens_to_ids("<|im_end|>"))
        # print(input_ids[0])

        attention_mask = attention_mask.to(self.device)

        out = self.LLM(
            feature_proj=self.feature_proj,
            input_ids=input_ids, 
            image_features=image_features,
            attention_mask=attention_mask,
            text_tokenizer=self.text_tokenizer
        )

        loss1 = out.loss
        # print(loss1)

        return CausalLMOutputWithPast(
            loss=loss1,
            logits=out.logits,
            past_key_values=out.past_key_values,
            hidden_states=out.hidden_states,
            attentions=out.attentions,
        )

      