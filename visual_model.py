import transformers
import torch
from typing import Optional
from torch import nn
from PIL import Image
from transformers import SiglipModel, SiglipVisionModel,SiglipVisionConfig
from transformers.utils import add_start_docstrings_to_model_forward


class VisualModel(SiglipVisionModel):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__(config)
        self.vision_embed_dim = config.hidden_size 
    def forward(
        self,
        pixel_values,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
    ):
        pixel_values = pixel_values.to(dtype=torch.bfloat16)

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )
        pooled_output = vision_outputs.last_hidden_state  # pooled_output
        return pooled_output
        




