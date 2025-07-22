# 从零开始搭建多模态大模型
## 1.简介
使用预训练的LLM与视觉编码器作为基础组件并进行适当改写，以支持多模态的输入；编写了projector层，将来自视觉编码器的图像特征映射到LLM支持的输入维度，实现模态对齐。
- 数据集：[minimind-v_dataset](https://www.modelscope.cn/datasets/gongjy/minimind-v_dataset/files"点击跳转")
- 视觉编码器：[google/siglip-base-patch16-224](https://hf-mirror.com/google/siglip-base-patch16-224"点击跳转")
- 语言模型：[Qwen/Qwen2.5-0.5B](https://hf-mirror.com/Qwen/Qwen2.5-0.5B"点击跳转")
## 2.实现流程
### 2.1. 数据处理
本项目使用的数据集质量较高，无需清洗，只需处理成对话格式即可。
原始数据如下：
```
{"conversations": [{"role": "user", "content": "绘制清晰简洁的图片摘要.\n<image>"}, {"role": "assistant", "content": "西拉穿着这件上衣和这件裙子看起来很漂亮,当时她和以前大学的同学一起表演"}], "image": "GCC_train_000000001.jpg"}
```
处理成如下的对话格式：
```
<|im_start|>user绘制清晰简洁的图片摘要.<|vision_start|><|vision_end|><|endoftext|><|im_end|>
<|im_start|>assistant西拉穿着这件上衣和这件裙子看起来很漂亮,当时她和以前大学的同学一起表演<|endoftext|><|im_end|>
```
<|im_start|>，<|im_end|>，<|endoftext|>，<|vision_start|>，<|vision_end|>，user，assistant均为特殊token，除user，assistant之外，其余均为分词器预训练的special tokens。后续将196个图像特征添加到<|im_start|>和<|im_end|>两个token之间。

图像数据通过预训练的image_processor处理成pixel_values，形状为（batch_size,channels,w,h）
### 2.2. 模型搭建
基本思路：

1. 编写一个VLM类，集成了视觉编码器、LLM、映射层，并包含了分词、图像文本拼接、标签构建、掩码构建等功能。
2. 取视觉编码器最后一个隐藏状态作为图像特征，张量形状为（batch_size,num_patches,hidden_dim），经过映射层后得到对齐的图像特征aligned_image_features.
3. 将文本进行分词得到input_ids，采取左填充方式，长度设置为32，之后与aligned_image_features一起输入LLM类的实例。
4. 在LLM类中，首先将input_ids进行嵌入，得到嵌入向量text_embeds;将text_embeds与aligned_image_features进行拼接，同时要改变padding_mask的长度，并手动构建因果掩码casual_mask,保证prompt部分不能看到答案，答案部分只能看到当前输出之前的内容。将padding_mask与causal_mask进行叠加，得到最终的combined_mask;此外，在LLM中还要实现labels的手动构建，保证序列中除了assistant token至<|im_end|>这段序列之外，其余部分均设置为-100。

