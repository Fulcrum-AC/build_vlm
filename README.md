# 从零开始搭建多模态大模型
## 1. 简介
使用预训练的LLM与视觉编码器作为基础组件并进行适当改写，以支持多模态的输入；编写了projector层，将来自视觉编码器的图像特征映射到LLM支持的输入维度，实现模态对齐。
- 数据集：[minimind-v_dataset](https://www.modelscope.cn/datasets/gongjy/minimind-v_dataset/files"点击跳转")
- 视觉编码器：[google/siglip-base-patch16-224](https://hf-mirror.com/google/siglip-base-patch16-224"点击跳转")
- 语言模型：[Qwen/Qwen2.5-0.5B](https://hf-mirror.com/Qwen/Qwen2.5-0.5B"点击跳转")
## 2. 实现流程
### 2.1. 环境搭建
采用如下命令安装项目所需依赖
```
# 安装PyTorch nightly (支持CUDA 12.8)
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# 安装所需的库
pip install transformers peft accelerate
```
### 2.2. 数据处理
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

图像数据通过预训练的image_processor处理成pixel_values，形状为（batch_size,channels,height,width）
### 2.3. 模型搭建
基本思路：

1. 编写一个VLM类，集成了视觉编码器、LLM、映射层，并包含了分词、图像文本拼接、标签构建、掩码构建等功能。
2. 取视觉编码器最后一个隐藏状态作为图像特征，张量形状为（batch_size,num_patches,hidden_dim），经过映射层后得到对齐的图像特征aligned_image_features.
3. 将文本进行分词得到input_ids，采取左填充方式，长度设置为32，之后与aligned_image_features一起输入LLM类的实例。
4. 在LLM类中，首先将input_ids进行嵌入，得到嵌入向量text_embeds;将text_embeds与aligned_image_features进行拼接，同时要改变padding_mask的长度，并手动构建因果掩码casual_mask,保证prompt部分不能看到答案，答案部分只能看到当前输出之前的内容。将padding_mask与causal_mask进行叠加，得到最终的combined_mask;此外，在LLM中还要实现labels的手动构建，保证序列中除了assistant token至<|im_end|>这段序列之外，其余部分均设置为-100。
### 2.4. 微调
冻结视觉编码器，并对映射层进行全量微调，对LLM以LoRA方式进行参数高效微调。

训练时的参数设置如下：
```python
train_args = TrainingArguments(
        output_dir="./vlm_model_output5",
        per_device_train_batch_size=40,  # 设置训练批大小
        save_strategy="steps",        # 按步数保存
        save_steps=500,               # 每500步保存一次检查点
        save_total_limit=20,           # 只保留最新的20个检查点
        logging_strategy="steps",     # 按步数记录日志
        logging_steps=10,            # 每10步记录一次日志
        bf16=True, 
        learning_rate=3e-4,  
        num_train_epochs=8,
        weight_decay=0.01,    # 添加权重衰减防止过拟合
        warmup_ratio=0.02,    # 总步数的2%用于预热
        lr_scheduler_type="cosine",  # 余弦退火
        max_grad_norm=1.0,     # 梯度裁剪
        save_safetensors=False # 禁用Safetensors格式
    )
```
## 3. 测试效果

<table>
  <thead>
    <tr>
      <th>图片</th>
      <th>描述</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>
        <img src="./eval_images/GCC_train_000000000.jpg">
      </td>
      <td>1.在大雨和大风中,在城市街道上,有大量汽车和行人。<br> 2.拥挤的交通在城市中心的街道上。</td>
    </tr>
    <tr>
      <td>
        <img src="./eval_images/train-00000-of-00001_image_43_0.jpg">
      </td>
      <td>1.一个花瓶里有黄色的花,上面有紫色的叶子。<br> 2.花瓶和花束,有黄色和粉红色的花</td>
    </tr>
    <tr>
      <td>
        <img src="./eval_images/GCC_train_000000001.jpg">
      </td>
      <td>1.女歌手在舞台上表演。<br> 2.在我们参加的时装周的服装秀上,一个人穿着黑色的裙子和一件白色的衬衫。</td>
    </tr>
    <tr>
      <td>
        <img src="./eval_images/GCC_train_000000023.jpg">
      </td>
      <td>1.一个大钱包,里面装满了钱,用一个大锁盖在白色背景上隔离。<br> 2.将金钱和现金放入保险箱,防止被盗或丢失。</td>
    </tr>     
    <tr> 
      <td>
        <img src="./eval_images/GCC_train_000000025.jpg">
      </td>
      <td>1.当您在球场上打高尔夫球时,您会看到一个有趣的景观。。<br> 2.在高尔夫球场的右前方,有起伏的地形。</td>
    </tr>  
  </tbody>
</table>
